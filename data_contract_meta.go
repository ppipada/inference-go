package inference

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
)

// DataContractVersion is bumped when the *schema* of the contract types changes.
const DataContractVersion = "v1.0.0"

// DataContractFiles lists files that define the data contract.
// Paths are relative to the repo root.
//
// These files should only contain data-contract types (structs/enums) that
// downstream consumers rely on structurally. Any change to these files will
// change the contract hash. It does NOT contain api contracts.
var DataContractFiles = []string{
	"spec/data_cache.go",
	"spec/data_citation.go",
	"spec/data_content.go",
	"spec/data_error.go",
	"spec/data_io_union.go",
	"spec/data_model.go",
	"spec/data_tool.go",
}

// DataContractHash is a SHA-256 of the contents of DataContractFiles.
// It is validated by tests and can be used by downstream consumers to check
// that they are running against the contract version they were built for.
//
// Format: "sha256:<hexstring>".
const DataContractHash = "sha256:855faa3568461dc5ab8fff0de61a90bcac1b602bc84ded814cc49aa05b8cb108"

// DataContractInfo is the public shape returned to callers who want to
// validate they are compatible with this version of the contract.
type DataContractInfo struct {
	Version string   `json:"version"`
	Hash    string   `json:"hash"`
	Files   []string `json:"files"`
}

// GetDataContractInfo returns the current contract version/hash metadata.
func GetDataContractInfo() DataContractInfo {
	return DataContractInfo{
		Version: DataContractVersion,
		Hash:    DataContractHash,
		Files:   append([]string(nil), DataContractFiles...),
	}
}

// ComputeDataContractHash recomputes the SHA-256 hash of the contract files'
// contents. It is intended for use in tests and development tooling.
//
// NOTE: This function assumes it is run in a source checkout of the module
// where the paths in DataContractFiles exist on disk. It is not suitable for
// use in production binaries where the Go source tree might not be available.
func ComputeDataContractHash() (string, error) {
	h := sha256.New()

	for _, rel := range DataContractFiles {
		// Paths are relative to module root (where "spec" lives).
		path := filepath.FromSlash(rel)

		data, err := os.ReadFile(path)
		if err != nil {
			return "", fmt.Errorf("read data contract file %q: %w", path, err)
		}

		if _, err := h.Write(data); err != nil {
			return "", fmt.Errorf("hash data contract file %q: %w", path, err)
		}

		// Separator for determinism.
		if _, err := h.Write([]byte("\n")); err != nil {
			return "", fmt.Errorf("hash separator: %w", err)
		}
	}

	return "sha256:" + hex.EncodeToString(h.Sum(nil)), nil
}

// ValidateDataContract recomputes the hash and compares it to DataContractHash.
// Tests in this module should call this to enforce that any schema change in
// the contract files is accompanied by an explicit update of DataContractHash
// (and, if breaking, DataContractVersion).
func ValidateDataContract() error {
	computed, err := ComputeDataContractHash()
	if err != nil {
		return err
	}
	if computed != DataContractHash {
		return fmt.Errorf(
			"data contract hash mismatch: compiled=%s, computed=%s. If this change is intentional, update DataContractHash in data_contract.go and bump DataContractVersion",
			DataContractHash,
			computed,
		)
	}
	return nil
}
