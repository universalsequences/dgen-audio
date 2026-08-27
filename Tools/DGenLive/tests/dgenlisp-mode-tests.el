;;; dgenlisp-mode-tests.el --- Tests for dgenlisp-mode -*- lexical-binding: t; -*-
(require 'ert)
(require 'dgenlisp-mode)

(ert-deftest dgenlisp-mode-activates-and-completes ()
  (with-temp-buffer
    (dgenlisp-mode)
    (should (derived-mode-p 'scheme-mode))
    (insert "pha")
    (let ((completion (dgenlisp-completion-at-point)))
      (should (equal (nth 2 completion) dgenlisp--forms))
      (should (member "phasor" (all-completions "pha" (nth 2 completion)))))
    (erase-buffer)
    (insert "@def")
    (let ((completion (dgenlisp-completion-at-point)))
      (should (member "@default" (all-completions "@def" (nth 2 completion)))))))

(ert-deftest dgenlisp-server-command-round-trip ()
  (let* ((socket (make-temp-name "/tmp/dgen-live-elisp-test-"))
         (server (make-network-process
                  :name "dgen-live-test-server" :family 'local :service socket
                  :server t :noquery t
                  :log (lambda (_server client _message)
                         (set-process-filter
                          client (lambda (process _input)
                                   (process-send-string process "OK pong\n")))))))
    (unwind-protect
        (let ((dgenlisp-server-socket socket))
          (should (equal (dgenlisp--server-command "PING") "OK pong")))
      (delete-process server)
      (when (file-exists-p socket) (delete-file socket)))))

(provide 'dgenlisp-mode-tests)
