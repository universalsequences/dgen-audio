;;; dgenlisp-mode.el --- Live coding for DGenLisp -*- lexical-binding: t; -*-

;; A small major mode and client for Tools/DGenLive/dgen-live.

(require 'scheme)
(require 'subr-x)

(defgroup dgenlisp nil "Edit and live-evaluate DGenLisp programs." :group 'languages)

(defcustom dgenlisp-compiler "dgenlisp"
  "Path to the DGenLisp compiler executable."
  :type 'file)

(defcustom dgenlisp-server-socket
  (format "/tmp/dgen-live-%d.sock" (user-uid))
  "Unix-domain socket used by dgen-live."
  :type 'file)

(defcustom dgenlisp-sample-rate 48000
  "Sample rate passed to DGenLisp when evaluating a buffer."
  :type 'integer)

(defcustom dgenlisp-max-frames 512
  "Maximum frame count compiled into a live patch.
This must be at least the server's --block-size."
  :type 'integer)

(defcustom dgenlisp-artifact-directory
  (expand-file-name "dgen-live" temporary-file-directory)
  "Parent directory for live compilation artifacts."
  :type 'directory)

(defconst dgenlisp--forms
  '("%" "*" "+" "-" "/" "abs" "accum" "atan" "atan2"
    "audio-tensor" "biquad" "buffer" "ceil" "click" "clip"
    "complex-conj" "complex-mul" "compressor" "conv1d" "conv2d" "cos"
    "cumsum" "def" "defmacro" "delay" "e" "eq" "exp" "expand" "false"
    "fft" "floor" "full" "gather" "gswitch" "gt" "gte" "hann" "hop-hold"
    "ifft" "in" "iota" "ir" "latch" "log" "log10" "lt" "lte"
    "make-history" "make-tensor-history" "matmul" "max" "max-axis" "mean"
    "mean-axis" "min" "mix" "mse" "noise" "ones" "out" "overlap-add"
    "pad" "param" "partitioned-convolve" "partitioned-spectral-mac"
    "partition-ir" "peek" "peek-row" "phase-vocoder" "phasor" "pi"
    "polar-fft" "pow" "ramp2trig" "randn" "read-history"
    "read-tensor-history" "rect-fft" "relu" "repeat" "reshape" "round"
    "sample" "scale" "selector" "shrink" "sigmoid" "sign" "sin" "softmax"
    "spectrum-delay" "spectrum-delay-mod" "sqrt" "stateful-phasor" "sum"
    "sum-axis" "svf-freq" "tan" "tanh" "tensor" "tensor-param" "to-signal"
    "transpose" "triangle" "true" "tuple" "twopi" "wavetable"
    "wavetable-param" "window" "windows" "wrap" "write-history"
    "write-tensor-history" "zeros"
    "@attack" "@axes" "@axis" "@cutoff" "@data" "@default"
    "@default-file" "@env" "@file" "@gain" "@group" "@knee" "@max"
    "@max-frames" "@min" "@mod" "@mode" "@mod-mode" "@modulator"
    "@name" "@padding" "@q" "@ranges" "@ratio" "@release" "@repeats"
    "@role" "@shape" "@sidechain" "@threshold" "@unit")
  "DGenLisp forms offered for completion and highlighting.")

(defun dgenlisp-completion-at-point ()
  "Complete a DGenLisp form at point."
  (let ((end (point))
        (start (save-excursion
                 (skip-chars-backward "[:alnum:]_@%+*/<>=!?-" )
                 (point))))
    (when (< start end)
      (list start end dgenlisp--forms :exclusive 'no))))

(defun dgenlisp--server-command (command)
  "Send COMMAND to dgen-live and return its one-line response."
  (unless (file-exists-p dgenlisp-server-socket)
    (user-error "DGen server socket does not exist: %s" dgenlisp-server-socket))
  (let* ((buffer (generate-new-buffer " *dgen-live-response*"))
         (process (condition-case error
                      (make-network-process
                       :name "dgen-live-client" :family 'local
                       :service dgenlisp-server-socket :buffer buffer
                       :coding 'utf-8-unix :noquery t)
                    (error (kill-buffer buffer) (signal (car error) (cdr error)))))
         (deadline (+ (float-time) 5.0))
         response)
    (unwind-protect
        (progn
          (process-send-string process (concat command "\n"))
          (while (and (< (float-time) deadline)
                      (progn
                        (with-current-buffer buffer
                          (goto-char (point-min))
                          (not (search-forward "\n" nil t)))))
            (accept-process-output process 0.05))
          (with-current-buffer buffer
            (setq response (string-trim (buffer-string))))
          (when (string-empty-p response)
            (error "Timed out waiting for dgen-live"))
          response)
      (when (process-live-p process) (delete-process process))
      (kill-buffer buffer))))

(defun dgenlisp-eval-buffer ()
  "Compile the current buffer and hot-swap it into dgen-live."
  (interactive)
  ;; Generated C kernels process four float lanes at a time.
  (unless (and (> dgenlisp-max-frames 0) (zerop (% dgenlisp-max-frames 4)))
    (user-error "dgenlisp-max-frames must be a positive multiple of 4 for SIMD"))
  (let* ((stamp (format "%d-%06d" (truncate (* 1000 (float-time))) (random 1000000)))
         (name (concat "patch-" stamp))
         (directory (expand-file-name name dgenlisp-artifact-directory))
         (manifest (expand-file-name (concat name ".json") directory))
         (asset-base (if buffer-file-name
                         (file-name-directory buffer-file-name)
                       default-directory))
         (output-buffer (get-buffer-create "*DGenLisp Compile*")))
    (make-directory directory t)
    (with-current-buffer output-buffer
      (let ((inhibit-read-only t)) (erase-buffer)))
    (let ((status
           (apply #'call-process-region
                  (point-min) (point-max) dgenlisp-compiler nil
                  (list output-buffer t) nil
                  (list "compile" "-" "-o" directory "--name" name
                        "--sample-rate" (number-to-string dgenlisp-sample-rate)
                        "--max-frames" (number-to-string dgenlisp-max-frames)
                        "--asset-base" asset-base))))
      (unless (and (integerp status) (zerop status))
        (display-buffer output-buffer)
        (error "DGenLisp compilation failed (exit %s)" status)))
    (let ((response (dgenlisp--server-command (concat "LOAD " manifest))))
      (unless (string-prefix-p "OK " response)
        (display-buffer output-buffer)
        (error "dgen-live rejected patch: %s" response))
      (message "DGenLisp hot-swapped: %s" response))))

(defun dgenlisp-stop ()
  "Replace the current live patch with silence."
  (interactive)
  (let ((response (dgenlisp--server-command "STOP")))
    (if (string-prefix-p "OK " response) (message "%s" response)
      (error "%s" response))))

(defvar dgenlisp-mode-map
  (let ((map (make-sparse-keymap)))
    (set-keymap-parent map scheme-mode-map)
    (define-key map (kbd "C-c C-c") #'dgenlisp-eval-buffer)
    (define-key map (kbd "C-c C-s") #'dgenlisp-stop)
    map))

;;;###autoload
(define-derived-mode dgenlisp-mode scheme-mode "DGenLisp"
  "Major mode for DGenLisp with whole-buffer live evaluation."
  (setq-local comment-start ";")
  (setq-local comment-end "")
  (add-hook 'completion-at-point-functions #'dgenlisp-completion-at-point nil t)
  (font-lock-add-keywords
   nil `((,(regexp-opt dgenlisp--forms 'symbols) . font-lock-builtin-face))))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.lisp\\'" . dgenlisp-mode))

(provide 'dgenlisp-mode)
;;; dgenlisp-mode.el ends here
