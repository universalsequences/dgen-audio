#define _GNU_SOURCE
#include <alsa/asoundlib.h>
#include <dlfcn.h>
#include <errno.h>
#include <json-c/json.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include "../../toolchain/include/dgen_runtime.h"

extern const DGenHostServicesV1 *dgen_reference_host_services_v1(void);

typedef void (*DGenProcessFn)(const float *const *, float *const *, uint32_t,
                              void *, const DGenProcessContextV1 *,
                              const DGenHostServicesV1 *);

enum { DGEN_LIVE_MAX_CHANNELS = 64 };

typedef struct Node {
  void *handle;
  DGenProcessFn process;
  float *state;
  size_t state_slots;
  unsigned input_channels;
  unsigned output_channels;
  unsigned max_frames;
  char *manifest_path;
  struct Node *retired_next;
} Node;

typedef struct {
  _Atomic(Node *) current;
  Node *retired;
  unsigned sample_rate;
  unsigned block_size;
  unsigned latency_us;
  const char *device;
  int no_audio;
  _Atomic int running;
  _Atomic unsigned long xruns;
  _Atomic unsigned long partial_writes;
  pthread_t audio_thread;
} Server;

static volatile sig_atomic_t caught_signal = 0;
static void on_signal(int sig) { (void)sig; caught_signal = 1; }

static int json_int(struct json_object *object, const char *key, int *out) {
  struct json_object *value = NULL;
  if (!json_object_object_get_ex(object, key, &value) ||
      !json_object_is_type(value, json_type_int))
    return 0;
  *out = json_object_get_int(value);
  return 1;
}

static unsigned channel_count(struct json_object *manifest, const char *key) {
  struct json_object *array = NULL;
  unsigned count = 1;
  if (!json_object_object_get_ex(manifest, key, &array) ||
      !json_object_is_type(array, json_type_array))
    return count;
  size_t length = json_object_array_length(array);
  for (size_t i = 0; i < length; ++i) {
    struct json_object *entry = json_object_array_get_idx(array, i);
    int channel = 0;
    if (entry && json_int(entry, "channel", &channel) && channel >= 0 &&
        (unsigned)channel + 1 > count)
      count = (unsigned)channel + 1;
  }
  return count;
}

static int initialize_state(Node *node, struct json_object *manifest,
                            char *error, size_t error_size) {
  struct json_object *array = NULL;
  if (json_object_object_get_ex(manifest, "tensorInitData", &array)) {
    if (!json_object_is_type(array, json_type_array)) {
      snprintf(error, error_size, "tensorInitData is not an array");
      return 0;
    }
    size_t count = json_object_array_length(array);
    for (size_t i = 0; i < count; ++i) {
      struct json_object *entry = json_object_array_get_idx(array, i);
      struct json_object *data = NULL;
      int offset = -1;
      if (!entry || !json_int(entry, "offset", &offset) || offset < 0 ||
          !json_object_object_get_ex(entry, "data", &data) ||
          !json_object_is_type(data, json_type_array)) {
        snprintf(error, error_size, "invalid tensorInitData entry %zu", i);
        return 0;
      }
      size_t values = json_object_array_length(data);
      if ((size_t)offset > node->state_slots ||
          values > node->state_slots - (size_t)offset) {
        snprintf(error, error_size, "tensorInitData entry %zu exceeds state", i);
        return 0;
      }
      for (size_t j = 0; j < values; ++j) {
        struct json_object *value = json_object_array_get_idx(data, j);
        if (!value || (!json_object_is_type(value, json_type_double) &&
                       !json_object_is_type(value, json_type_int))) {
          snprintf(error, error_size, "non-numeric tensor value at entry %zu", i);
          return 0;
        }
        node->state[offset + j] = (float)json_object_get_double(value);
      }
    }
  }

  if (json_object_object_get_ex(manifest, "params", &array)) {
    if (!json_object_is_type(array, json_type_array)) {
      snprintf(error, error_size, "params is not an array");
      return 0;
    }
    size_t count = json_object_array_length(array);
    for (size_t i = 0; i < count; ++i) {
      struct json_object *entry = json_object_array_get_idx(array, i);
      struct json_object *value = NULL;
      int cell = -1, span = 1;
      if (!entry || !json_int(entry, "cellId", &cell) || cell < -1 ||
          !json_object_object_get_ex(entry, "default", &value) ||
          (!json_object_is_type(value, json_type_double) &&
           !json_object_is_type(value, json_type_int))) {
        snprintf(error, error_size, "invalid params entry %zu", i);
        return 0;
      }
      (void)json_int(entry, "cellSpan", &span);
      if (cell == -1) continue; /* Declared but unreachable parameter. */
      if (span < 1 || (size_t)cell > node->state_slots ||
          (size_t)span > node->state_slots - (size_t)cell) {
        snprintf(error, error_size, "params entry %zu exceeds state", i);
        return 0;
      }
      float initial = (float)json_object_get_double(value);
      for (int j = 0; j < span; ++j) node->state[cell + j] = initial;
    }
  }
  return 1;
}

static void destroy_node(Node *node) {
  if (!node) return;
  free(node->manifest_path);
  free(node->state);
  if (node->handle) dlclose(node->handle);
  free(node);
}

static Node *load_node(const char *manifest_path, unsigned block_size,
                       char *error, size_t error_size) {
  struct json_object *manifest = json_object_from_file(manifest_path);
  if (!manifest) {
    snprintf(error, error_size, "cannot parse manifest: %s", manifest_path);
    return NULL;
  }

  Node *node = calloc(1, sizeof(*node));
  int version = 0, slots = 0, max_frames = 0;
  struct json_object *abi = NULL, *dylib = NULL;
  if (!node || !json_int(manifest, "version", &version) || version != 3 ||
      !json_int(manifest, "totalMemorySlots", &slots) || slots < 0 ||
      !json_int(manifest, "maxFrameCount", &max_frames) || max_frames < 1 ||
      !json_object_object_get_ex(manifest, "processAbi", &abi) ||
      strcmp(json_object_get_string(abi), "dgen-host-abi-v1") != 0 ||
      !json_object_object_get_ex(manifest, "dylib", &dylib)) {
    snprintf(error, error_size, "unsupported or incomplete DGen manifest");
    goto fail;
  }
  if ((unsigned)max_frames < block_size) {
    snprintf(error, error_size, "patch maxFrameCount %d is smaller than block size %u",
             max_frames, block_size);
    goto fail;
  }

  char manifest_real[PATH_MAX], dylib_path[PATH_MAX], directory[PATH_MAX];
  if (!realpath(manifest_path, manifest_real)) {
    snprintf(error, error_size, "realpath(%s): %s", manifest_path, strerror(errno));
    goto fail;
  }
  snprintf(directory, sizeof(directory), "%s", manifest_real);
  char *slash = strrchr(directory, '/');
  if (!slash) goto fail;
  *slash = '\0';
  const char *dylib_name = json_object_get_string(dylib);
  if (!dylib_name || strchr(dylib_name, '/')) {
    snprintf(error, error_size, "manifest dylib must be a file name");
    goto fail;
  }
  if (snprintf(dylib_path, sizeof(dylib_path), "%s/%s", directory, dylib_name) >=
      (int)sizeof(dylib_path)) {
    snprintf(error, error_size, "dylib path is too long");
    goto fail;
  }

  node->handle = dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL);
  if (!node->handle) {
    snprintf(error, error_size, "dlopen: %s", dlerror());
    goto fail;
  }
  void *symbol = dlsym(node->handle, "dgen_process_v1");
  if (!symbol) {
    snprintf(error, error_size, "dlsym(dgen_process_v1): %s", dlerror());
    goto fail;
  }
  memcpy(&node->process, &symbol, sizeof(node->process));
  node->state_slots = (size_t)(slots < 1024 ? 1024 : slots);
  node->state = calloc(node->state_slots, sizeof(float));
  node->manifest_path = strdup(manifest_real);
  node->input_channels = channel_count(manifest, "inputs");
  node->output_channels = channel_count(manifest, "outputs");
  node->max_frames = (unsigned)max_frames;
  if (node->input_channels > DGEN_LIVE_MAX_CHANNELS ||
      node->output_channels > DGEN_LIVE_MAX_CHANNELS) {
    snprintf(error, error_size, "patch exceeds the %d-channel host limit",
             DGEN_LIVE_MAX_CHANNELS);
    goto fail;
  }
  if (!node->state || !node->manifest_path) {
    snprintf(error, error_size, "out of memory");
    goto fail;
  }
  if (!initialize_state(node, manifest, error, error_size)) goto fail;
  json_object_put(manifest);
  return node;

fail:
  json_object_put(manifest);
  destroy_node(node);
  return NULL;
}

static void retire_and_swap(Server *server, Node *node) {
  Node *old = atomic_exchange_explicit(&server->current, node, memory_order_acq_rel);
  if (old) {
    old->retired_next = server->retired;
    server->retired = old;
  }
}

static int render_node(Node *node, unsigned frames, unsigned sample_rate,
                       float ***outputs_out) {
  unsigned ins = node ? node->input_channels : 1;
  unsigned outs = node ? node->output_channels : 1;
  float **inputs = calloc(ins, sizeof(*inputs));
  float **outputs = calloc(outs, sizeof(*outputs));
  if (!inputs || !outputs) goto fail;
  for (unsigned c = 0; c < ins; ++c)
    if (!(inputs[c] = calloc(frames, sizeof(float)))) goto fail;
  for (unsigned c = 0; c < outs; ++c)
    if (!(outputs[c] = calloc(frames, sizeof(float)))) goto fail;
  if (node) {
    DGenProcessContextV1 context = {
      .abi_version = DGEN_ABI_VERSION_V1,
      .struct_size = sizeof(DGenProcessContextV1),
      .sample_rate = (float)sample_rate,
      .reserved = 0,
    };
    node->process((const float *const *)inputs, outputs, frames, node->state,
                  &context, dgen_reference_host_services_v1());
  }
  for (unsigned c = 0; c < ins; ++c) free(inputs[c]);
  free(inputs);
  *outputs_out = outputs;
  return (int)outs;
fail:
  if (inputs) for (unsigned c = 0; c < ins; ++c) free(inputs[c]);
  if (outputs) for (unsigned c = 0; c < outs; ++c) free(outputs[c]);
  free(inputs); free(outputs);
  return 0;
}

static void free_outputs(float **outputs, unsigned channels) {
  for (unsigned c = 0; c < channels; ++c) free(outputs[c]);
  free(outputs);
}

static int pcm_write_all(Server *server, snd_pcm_t *pcm, const float *samples,
                         snd_pcm_uframes_t frames) {
  snd_pcm_uframes_t offset = 0;
  while (offset < frames && atomic_load_explicit(&server->running, memory_order_relaxed)) {
    snd_pcm_sframes_t written = snd_pcm_writei(pcm, samples + offset * 2,
                                               frames - offset);
    if (written == -EAGAIN) {
      snd_pcm_wait(pcm, 1000);
      continue;
    }
    if (written < 0) {
      atomic_fetch_add_explicit(&server->xruns, 1, memory_order_relaxed);
      int recovered = snd_pcm_recover(pcm, (int)written, 1);
      if (recovered < 0) return recovered;
      continue; /* Recovery prepared the PCM; retry samples we have not written. */
    }
    if (written == 0) continue;
    if ((snd_pcm_uframes_t)written < frames - offset)
      atomic_fetch_add_explicit(&server->partial_writes, 1, memory_order_relaxed);
    offset += (snd_pcm_uframes_t)written;
  }
  return offset == frames ? 0 : -ECANCELED;
}

static void *audio_main(void *opaque) {
  Server *server = opaque;
  snd_pcm_t *pcm = NULL;
  int rc = snd_pcm_open(&pcm, server->device, SND_PCM_STREAM_PLAYBACK, 0);
  if (rc < 0) {
    fprintf(stderr, "dgen-live: ALSA open %s: %s\n", server->device, snd_strerror(rc));
    atomic_store(&server->running, 0);
    return NULL;
  }
  rc = snd_pcm_set_params(pcm, SND_PCM_FORMAT_FLOAT_LE,
                          SND_PCM_ACCESS_RW_INTERLEAVED, 2, server->sample_rate,
                          1, server->latency_us);
  if (rc < 0) {
    fprintf(stderr, "dgen-live: ALSA setup: %s\n", snd_strerror(rc));
    snd_pcm_close(pcm);
    atomic_store(&server->running, 0);
    return NULL;
  }
  size_t channel_samples = DGEN_LIVE_MAX_CHANNELS * server->block_size;
  float *input_storage = calloc(channel_samples, sizeof(float));
  float *output_storage = calloc(channel_samples, sizeof(float));
  float *interleaved = malloc(server->block_size * 2 * sizeof(float));
  float *inputs[DGEN_LIVE_MAX_CHANNELS];
  float *outputs[DGEN_LIVE_MAX_CHANNELS];
  if (!input_storage || !output_storage || !interleaved) {
    free(input_storage); free(output_storage); free(interleaved);
    snd_pcm_close(pcm);
    atomic_store(&server->running, 0);
    return NULL;
  }
  for (unsigned c = 0; c < DGEN_LIVE_MAX_CHANNELS; ++c) {
    inputs[c] = input_storage + c * server->block_size;
    outputs[c] = output_storage + c * server->block_size;
  }
  DGenProcessContextV1 context = {
    .abi_version = DGEN_ABI_VERSION_V1,
    .struct_size = sizeof(DGenProcessContextV1),
    .sample_rate = (float)server->sample_rate,
    .reserved = 0,
  };
  while (atomic_load_explicit(&server->running, memory_order_relaxed)) {
    Node *node = atomic_load_explicit(&server->current, memory_order_acquire);
    unsigned channels = node ? node->output_channels : 1;
    memset(output_storage, 0, channel_samples * sizeof(float));
    if (node)
      node->process((const float *const *)inputs, outputs, server->block_size,
                    node->state, &context, dgen_reference_host_services_v1());
    for (unsigned i = 0; i < server->block_size; ++i) {
      float left = outputs[0][i];
      float right = channels > 1 ? outputs[1][i] : left;
      interleaved[i * 2] = left;
      interleaved[i * 2 + 1] = right;
    }
    if (pcm_write_all(server, pcm, interleaved, server->block_size) < 0) {
      atomic_store(&server->running, 0);
      break;
    }
  }
  free(input_storage);
  free(output_storage);
  free(interleaved);
  snd_pcm_drop(pcm);
  snd_pcm_close(pcm);
  return NULL;
}

static void reply_render(FILE *client, Server *server, unsigned frames) {
  Node *node = atomic_load_explicit(&server->current, memory_order_acquire);
  if (!node) { fprintf(client, "ERR no patch loaded\n"); return; }
  if (!server->no_audio) { fprintf(client, "ERR RENDER requires --no-audio\n"); return; }
  if (!frames || frames > node->max_frames || frames % 4 != 0) {
    fprintf(client, "ERR frame count must be divisible by 4 and at most %u\n",
            node->max_frames);
    return;
  }
  float **outputs = NULL;
  int channels = render_node(node, frames, server->sample_rate, &outputs);
  if (!channels) { fprintf(client, "ERR render allocation failed\n"); return; }
  fprintf(client, "OK [");
  for (unsigned i = 0; i < frames; ++i)
    fprintf(client, "%s%.9g", i ? "," : "", outputs[0][i]);
  fprintf(client, "]\n");
  free_outputs(outputs, (unsigned)channels);
}

static int serve_client(int fd, Server *server) {
  FILE *client = fdopen(fd, "r+");
  if (!client) { close(fd); return 1; }
  setvbuf(client, NULL, _IOLBF, 0);
  char *line = NULL;
  size_t capacity = 0;
  int keep_running = 1;
  while (keep_running && getline(&line, &capacity, client) >= 0) {
    line[strcspn(line, "\r\n")] = '\0';
    if (!strncmp(line, "LOAD ", 5) && line[5]) {
      char error[512];
      Node *node = load_node(line + 5, server->block_size, error, sizeof(error));
      if (!node) fprintf(client, "ERR %s\n", error);
      else {
        retire_and_swap(server, node);
        fprintf(client, "OK loaded %s\n", node->manifest_path);
      }
    } else if (!strcmp(line, "STOP")) {
      retire_and_swap(server, NULL);
      fprintf(client, "OK stopped\n");
    } else if (!strcmp(line, "PING")) {
      fprintf(client, "OK pong\n");
    } else if (!strcmp(line, "STATUS")) {
      fprintf(client, "OK {\"xruns\":%lu,\"partialWrites\":%lu}\n",
              atomic_load_explicit(&server->xruns, memory_order_relaxed),
              atomic_load_explicit(&server->partial_writes, memory_order_relaxed));
    } else if (!strncmp(line, "RENDER ", 7)) {
      char *end = NULL;
      unsigned long frames = strtoul(line + 7, &end, 10);
      if (!end || *end || frames > UINT32_MAX) fprintf(client, "ERR invalid frame count\n");
      else reply_render(client, server, (unsigned)frames);
    } else if (!strcmp(line, "QUIT")) {
      fprintf(client, "OK quitting\n");
      atomic_store(&server->running, 0);
      keep_running = 0;
    } else {
      fprintf(client, "ERR expected LOAD path, STOP, PING, STATUS, RENDER n, or QUIT\n");
    }
  }
  free(line);
  fclose(client);
  return keep_running;
}

static void usage(const char *program) {
  fprintf(stderr, "usage: %s [--socket PATH] [--device ALSA_DEVICE] "
                  "[--sample-rate HZ] [--block-size N] [--latency-ms N] "
                  "[--no-audio]\n", program);
}

int main(int argc, char **argv) {
  const char *socket_path = NULL;
  char default_socket[PATH_MAX];
  Server server = {
    .sample_rate = 48000,
    .block_size = 512,
    .latency_us = 100000,
    .device = "default"
  };
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--socket") && i + 1 < argc) socket_path = argv[++i];
    else if (!strcmp(argv[i], "--device") && i + 1 < argc) server.device = argv[++i];
    else if (!strcmp(argv[i], "--sample-rate") && i + 1 < argc) server.sample_rate = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--block-size") && i + 1 < argc) server.block_size = (unsigned)strtoul(argv[++i], NULL, 10);
    else if (!strcmp(argv[i], "--latency-ms") && i + 1 < argc) {
      unsigned long latency_ms = strtoul(argv[++i], NULL, 10);
      if (latency_ms > 10000) { usage(argv[0]); return 2; }
      server.latency_us = (unsigned)latency_ms * 1000;
    }
    else if (!strcmp(argv[i], "--no-audio")) server.no_audio = 1;
    else { usage(argv[0]); return 2; }
  }
  if (!server.sample_rate || !server.block_size || !server.latency_us ||
      server.block_size % 4 != 0) {
    fprintf(stderr, "dgen-live: block size must be a positive multiple of 4\n");
    usage(argv[0]);
    return 2;
  }
  if (!socket_path) {
    snprintf(default_socket, sizeof(default_socket), "/tmp/dgen-live-%lu.sock", (unsigned long)getuid());
    socket_path = default_socket;
  }
  if (strlen(socket_path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) {
    fprintf(stderr, "dgen-live: socket path too long\n"); return 2;
  }

  signal(SIGINT, on_signal); signal(SIGTERM, on_signal); signal(SIGPIPE, SIG_IGN);
  int listener = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listener < 0) { perror("socket"); return 1; }
  struct sockaddr_un address = {.sun_family = AF_UNIX};
  snprintf(address.sun_path, sizeof(address.sun_path), "%s", socket_path);
  unlink(socket_path);
  if (bind(listener, (struct sockaddr *)&address, sizeof(address)) < 0 ||
      chmod(socket_path, 0600) < 0 || listen(listener, 4) < 0) {
    perror("dgen-live socket"); close(listener); unlink(socket_path); return 1;
  }
  atomic_store(&server.running, 1);
  if (!server.no_audio && pthread_create(&server.audio_thread, NULL, audio_main, &server)) {
    perror("pthread_create"); close(listener); unlink(socket_path); return 1;
  }
  fprintf(stderr, "dgen-live: listening on %s%s\n", socket_path,
          server.no_audio ? " (no audio)" : "");

  while (!caught_signal && atomic_load(&server.running)) {
    struct pollfd ready = {.fd = listener, .events = POLLIN};
    int poll_result = poll(&ready, 1, 200);
    if (poll_result < 0) { if (errno == EINTR) continue; perror("poll"); break; }
    if (poll_result == 0) continue;
    int fd = accept(listener, NULL, NULL);
    if (fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
    serve_client(fd, &server);
  }
  atomic_store(&server.running, 0);
  close(listener);
  if (!server.no_audio) pthread_join(server.audio_thread, NULL);
  unsigned long xruns = atomic_load_explicit(&server.xruns, memory_order_relaxed);
  unsigned long partials = atomic_load_explicit(&server.partial_writes, memory_order_relaxed);
  if (xruns || partials)
    fprintf(stderr, "dgen-live: audio stats: %lu xruns/recoveries, %lu partial writes\n",
            xruns, partials);
  unlink(socket_path);
  destroy_node(atomic_load(&server.current));
  while (server.retired) { Node *next = server.retired->retired_next; destroy_node(server.retired); server.retired = next; }
  return 0;
}
