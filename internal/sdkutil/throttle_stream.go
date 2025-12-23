package sdkutil

import (
	"strings"
	"sync"
	"time"

	"github.com/ppipada/inference-go/spec"
)

const (
	FlushInterval  = 256 * time.Millisecond
	FlushChunkSize = 1024
)

// NewBufferedStreamer returns two functions:
//   - write(chunk)  -> use this instead of onDataFlush
//   - flush()       -> call once when streaming is finished
func NewBufferedStreamer(
	onDataFlush func(string) error,
	flushInterval time.Duration,
	maxSize int,
) (write func(string) error, flush func()) {
	if flushInterval <= 0 {
		flushInterval = FlushInterval
	}
	if maxSize <= 0 {
		maxSize = FlushChunkSize
	}
	var mu sync.Mutex
	var buf strings.Builder
	ticker := time.NewTicker(flushInterval)
	done := make(chan struct{})

	// Background goroutine time-based flush.
	go func() {
		defer Recover("buffered streamer background flush panic")

		for {
			select {
			case <-ticker.C:
				mu.Lock()
				if buf.Len() > 0 {
					data := buf.String()
					buf.Reset()
					mu.Unlock()
					_ = onDataFlush(data)
				} else {
					mu.Unlock()
				}
			case <-done:
				ticker.Stop()
				return
			}
		}
	}()

	// Returns the wrapped write.
	write = func(chunk string) error {
		mu.Lock()
		buf.WriteString(chunk)
		over := buf.Len() >= maxSize
		if over {
			data := buf.String()
			buf.Reset()
			mu.Unlock()
			// Size-based flush.
			return onDataFlush(data)
		}
		mu.Unlock()
		return nil
	}

	var once sync.Once
	// Flush everything, stop ticker.
	flush = func() {
		once.Do(func() {
			close(done)
			mu.Lock()
			if buf.Len() > 0 {
				data := buf.String()
				buf.Reset()
				mu.Unlock()
				_ = onDataFlush(data)
				return
			}
			mu.Unlock()
		})
	}

	return write, flush
}

// SafeCallStreamHandler invokes the provided StreamHandler and converts any
// panic into an error while logging the panic details. This prevents user
// callbacks from crashing the streaming loop.
func SafeCallStreamHandler(handler spec.StreamHandler, event spec.StreamEvent) (err error) {
	if handler == nil {
		return nil
	}

	defer Recover("stream handler panic",
		"kind", event.Kind,
		"provider", event.Provider,
		"model", event.Model,
	)

	return handler(event)
}

// ResolvedStreamConfig is the fully-specified streaming configuration used by
// providers after applying sensible defaults.
type ResolvedStreamConfig struct {
	FlushInterval  time.Duration
	FlushChunkSize int
}

// ResolveStreamConfig converts optional FetchCompletionOptions into a concrete
// ResolvedStreamConfig, falling back to library defaults as needed.
func ResolveStreamConfig(opts *spec.FetchCompletionOptions) ResolvedStreamConfig {
	cfg := ResolvedStreamConfig{
		FlushInterval:  FlushInterval,
		FlushChunkSize: FlushChunkSize,
	}
	if opts == nil || opts.StreamConfig == nil {
		return cfg
	}

	if opts.StreamConfig.FlushIntervalMillis > 0 {
		cfg.FlushInterval = time.Duration(opts.StreamConfig.FlushIntervalMillis) * time.Millisecond
	}
	if opts.StreamConfig.FlushChunkSize > 0 {
		cfg.FlushChunkSize = opts.StreamConfig.FlushChunkSize
	}
	return cfg
}
