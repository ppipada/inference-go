package sdkutil

import (
	"runtime/debug"

	"github.com/ppipada/inference-go/internal/logutil"
)

// Recover logs a panic (if any) at error level and prevents it from bringing
// down the goroutine's caller. It does not modify any returned error.
func Recover(msg string, fields ...any) {
	if r := recover(); r != nil {
		fields := append(fields, "panic", r, "stack", string(debug.Stack()))
		logutil.Error(msg, fields...)
	}
}
