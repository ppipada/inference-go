package logutil

import "encoding/json"

func LogJSON(v any) {
	p, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		Error("failed to marshal json", "err", err)
		return
	}
	Info("inferenceGo log", "json", string(p))
}
