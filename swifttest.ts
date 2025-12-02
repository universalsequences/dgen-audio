#!/usr/bin/env bun

import { spawn } from "bun";

// ANSI color codes
const colors = {
  reset: "\x1b[0m",
  red: "\x1b[31m",
  green: "\x1b[32m",
  yellow: "\x1b[33m",
  blue: "\x1b[34m",
  magenta: "\x1b[35m",
  cyan: "\x1b[36m",
  dim: "\x1b[2m",
};

// Get command line arguments (excluding first two which are bun and script path)
const args = process.argv.slice(2);

// Build swift test command
const testCommand = ["swift", "test"];

// Add filter if test name provided
if (args.length > 0) {
  testCommand.push("--filter", args[0]);
}

console.log(`${colors.cyan}Running: ${testCommand.join(" ")}${colors.reset}\n`);

// Spawn the swift test process
const proc = spawn({
  cmd: testCommand,
  stdout: "pipe",
  stderr: "pipe",
});

// Process output line by line
const decoder = new TextDecoder();

// Read stdout
(async () => {
  for await (const chunk of proc.stdout) {
    const text = decoder.decode(chunk);
    const lines = text.split("\n");

    for (const line of lines) {
      if (!line.trim()) continue;

      // Color success lines
      if (
        line.includes("passed") ||
        line.includes("✔") ||
        line.match(/Executed \d+ test.*with 0 failures/)
      ) {
        console.log(`${colors.green}${line}${colors.reset}`);
      }
      // Color failure lines
      else if (
        line.includes("failed") ||
        line.includes("✗") ||
        line.includes("error:") ||
        line.includes("failures") && !line.includes("0 failures")
      ) {
        console.log(`${colors.red}${line}${colors.reset}`);
      }
      // Color test suite headers
      else if (line.includes("Test Suite")) {
        console.log(`${colors.cyan}${line}${colors.reset}`);
      }
      // Color test case lines
      else if (line.includes("Test Case")) {
        console.log(`${colors.blue}${line}${colors.reset}`);
      }
      // Color the "Executed" summary lines
      else if (line.includes("Executed")) {
        console.log(`${colors.green}${line}${colors.reset}`);
      }
      // Color warnings
      else if (line.includes("warning:")) {
        console.log(`${colors.yellow}${line}${colors.reset}`);
      }
      // Dim build output and other noise
      else if (
        line.includes("Building for") ||
        line.includes("Build complete") ||
        line.includes("[") && line.includes("]") && line.includes("Compiling") ||
        line.includes("Write") ||
        line.includes("Linking") ||
        line.includes("Emitting")
      ) {
        console.log(`${colors.dim}${line}${colors.reset}`);
      }
      // Hide the confusing Swift Testing "0 tests" line
      else if (line.includes("Test run with 0 tests passed")) {
        // Skip this line entirely - it's just Swift Testing finding no @Test macros
        continue;
      }
      // Hide the Swift Testing startup lines
      else if (
        line.includes("◇ Test run started") ||
        line.includes("↳ Testing Library") ||
        line.includes("↳ Target Platform")
      ) {
        // Skip these lines - they're from Swift Testing framework
        continue;
      }
      // Default: print as-is
      else {
        console.log(line);
      }
    }
  }
})();

// Read stderr
(async () => {
  for await (const chunk of proc.stderr) {
    const text = decoder.decode(chunk);
    const lines = text.split("\n");

    for (const line of lines) {
      if (!line.trim()) continue;

      // Most stderr is warnings or errors
      if (line.includes("error:")) {
        console.error(`${colors.red}${line}${colors.reset}`);
      } else if (line.includes("warning:")) {
        console.error(`${colors.yellow}${line}${colors.reset}`);
      } else {
        console.error(`${colors.dim}${line}${colors.reset}`);
      }
    }
  }
})();

// Wait for process to complete
const exitCode = await proc.exited;

// Print final summary
if (exitCode === 0) {
  console.log(`\n${colors.green}✓ Tests passed!${colors.reset}`);
} else {
  console.log(`\n${colors.red}✗ Tests failed with exit code ${exitCode}${colors.reset}`);
  process.exit(exitCode);
}
