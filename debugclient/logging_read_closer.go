package debugclient

import (
	"bytes"
	"io"
	"sync"

	"github.com/ppipada/inference-go/internal/logutil"
)

type loggingReadCloser struct {
	io.ReadCloser

	buf   *bytes.Buffer
	state *HTTPDebugState
	cfg   DebugConfig

	mu        sync.Mutex
	finalized bool // ensure we attach Data once, even if both Read(EOF) and Close() happen.
}

func (lc *loggingReadCloser) Read(p []byte) (int, error) {
	n, err := lc.ReadCloser.Read(p)
	if n > 0 {
		lc.buf.Write(p[:n])
	}
	// Many SDKs read until EOF but never call Close() on resp.Body.
	if err == io.EOF {
		lc.finalize()
	}
	return n, err
}

func (lc *loggingReadCloser) Close() error {
	err := lc.ReadCloser.Close()
	lc.finalize()
	return err
}

// finalize attaches the buffered response body to state.response.Data exactly
// once, and applies sanitization/redaction.
func (lc *loggingReadCloser) finalize() {
	lc.mu.Lock()
	if lc.finalized {
		lc.mu.Unlock()
		return
	}
	lc.finalized = true
	lc.mu.Unlock()

	if lc.state == nil || lc.state.ResponseDetails == nil {
		return
	}
	if lc.cfg.DisableResponseBody {
		// Should not normally be wrapped in this mode, but guard anyway.
		return
	}

	dataBytes := lc.buf.Bytes()
	if len(dataBytes) == 0 {
		return
	}

	// Process and redact body.
	lc.state.ResponseDetails.Data = sanitizeBodyForDebug(dataBytes, false, lc.cfg)

	if lc.cfg.LogToSlog {
		logutil.Debug("http_debug: response body raw", "body", string(dataBytes))
	}
}
