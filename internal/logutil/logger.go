package logutil

import (
	"context"
	"log/slog"
	"sync"
)

var (
	mu           sync.RWMutex
	globalLogger *slog.Logger
)

func init() {
	// Default to a no-op logger so the library is silent unless a logger
	// is explicitly installed by the caller.
	globalLogger = slog.New(slog.DiscardHandler)
}

//
// Slog-compatible top-level functions.
//

// Default returns the current process-wide logger, analogous to slog.Default.
func Default() *slog.Logger {
	mu.RLock()
	defer mu.RUnlock()
	return globalLogger
}

// SetDefault sets the process-wide logger, analogous to slog.SetDefault.
// For compatibility with slog, passing nil panics.
func SetDefault(logger *slog.Logger) {
	mu.Lock()
	defer mu.Unlock()

	if logger == nil {
		logger = slog.New(slog.DiscardHandler)
	}
	globalLogger = logger
}

// Debug logs at LevelDebug using the process-wide logger.
func Debug(msg string, args ...any) {
	Default().Debug(msg, args...)
}

// DebugContext logs at LevelDebug with context using the process-wide logger.
func DebugContext(ctx context.Context, msg string, args ...any) {
	Default().DebugContext(ctx, msg, args...)
}

// Info logs at LevelInfo using the process-wide logger.
func Info(msg string, args ...any) {
	Default().Info(msg, args...)
}

// InfoContext logs at LevelInfo with context using the process-wide logger.
func InfoContext(ctx context.Context, msg string, args ...any) {
	Default().InfoContext(ctx, msg, args...)
}

// Warn logs at LevelWarn using the process-wide logger.
func Warn(msg string, args ...any) {
	Default().Warn(msg, args...)
}

// WarnContext logs at LevelWarn with context using the process-wide logger.
func WarnContext(ctx context.Context, msg string, args ...any) {
	Default().WarnContext(ctx, msg, args...)
}

// Error logs at LevelError using the process-wide logger.
func Error(msg string, args ...any) {
	Default().Error(msg, args...)
}

// ErrorContext logs at LevelError with context using the process-wide logger.
func ErrorContext(ctx context.Context, msg string, args ...any) {
	Default().ErrorContext(ctx, msg, args...)
}

// Log logs at the given level using the process-wide logger.
// Signature is identical to slog.Log.
func Log(ctx context.Context, level slog.Level, msg string, args ...any) {
	Default().Log(ctx, level, msg, args...)
}

// LogAttrs logs at the given level with pre-built attributes using the
// process-wide logger. Signature is identical to slog.LogAttrs.
func LogAttrs(ctx context.Context, level slog.Level, msg string, attrs ...slog.Attr) {
	Default().LogAttrs(ctx, level, msg, attrs...)
}

// With returns a logger that includes the supplied key/value pairs as
// attributes, rooted at the process-wide logger.
// Signature is identical to slog.With.
func With(args ...any) *slog.Logger {
	return Default().With(args...)
}
