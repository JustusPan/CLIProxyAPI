package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
)

type openAIResponsesSSEValidator struct {
	pendingChunks [][]byte
}

func (v *openAIResponsesSSEValidator) AddChunk(chunk []byte) ([][]byte, error) {
	if len(chunk) == 0 {
		return nil, nil
	}

	if len(v.pendingChunks) == 0 {
		needMore, err := inspectOpenAIResponsesSSEChunk(chunk, false)
		if err != nil {
			return nil, err
		}
		if !needMore {
			return [][]byte{chunk}, nil
		}
		v.pendingChunks = append(v.pendingChunks, cloneBytes(chunk))
		return nil, nil
	}

	v.pendingChunks = append(v.pendingChunks, cloneBytes(chunk))
	needMore, err := inspectOpenAIResponsesSSEChunk(v.pendingBytes(), false)
	if err != nil {
		v.pendingChunks = nil
		return nil, err
	}
	if needMore {
		return nil, nil
	}

	release := make([][]byte, len(v.pendingChunks))
	copy(release, v.pendingChunks)
	v.pendingChunks = nil
	return release, nil
}

func (v *openAIResponsesSSEValidator) Finish() ([][]byte, error) {
	if len(v.pendingChunks) == 0 {
		return nil, nil
	}

	needMore, err := inspectOpenAIResponsesSSEChunk(v.pendingBytes(), true)
	if err != nil {
		v.pendingChunks = nil
		return nil, err
	}
	if needMore {
		err = invalidSSEDataJSONError(v.pendingBytes())
		v.pendingChunks = nil
		return nil, err
	}

	release := make([][]byte, len(v.pendingChunks))
	copy(release, v.pendingChunks)
	v.pendingChunks = nil
	return release, nil
}

func (v *openAIResponsesSSEValidator) pendingBytes() []byte {
	if len(v.pendingChunks) == 0 {
		return nil
	}
	total := 0
	for _, chunk := range v.pendingChunks {
		total += len(chunk)
	}
	combined := make([]byte, 0, total)
	for _, chunk := range v.pendingChunks {
		combined = append(combined, chunk...)
	}
	return combined
}

func inspectOpenAIResponsesSSEChunk(chunk []byte, eof bool) (bool, error) {
	s := chunk
	for len(s) > 0 {
		line := s
		lineComplete := false
		if i := bytes.IndexByte(s, '\n'); i >= 0 {
			line = s[:i]
			s = s[i+1:]
			lineComplete = true
		} else {
			s = nil
		}

		line = bytes.TrimSpace(bytes.TrimRight(line, "\r"))
		if len(line) == 0 || !bytes.HasPrefix(line, []byte("data:")) {
			continue
		}

		data := bytes.TrimSpace(line[len("data:"):])
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) || json.Valid(data) {
			continue
		}

		if !lineComplete && !eof {
			return true, nil
		}
		return false, invalidSSEDataJSONError(data)
	}
	return false, nil
}

func invalidSSEDataJSONError(data []byte) error {
	const max = 512
	preview := data
	if len(preview) > max {
		preview = preview[:max]
	}
	return fmt.Errorf("invalid SSE data JSON (len=%d): %q", len(data), preview)
}
