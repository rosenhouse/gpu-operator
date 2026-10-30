//go:build linux

/**
# Copyright (c) NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
**/

package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// stubNvidiaSMI puts an nvidia-smi that runs script first on PATH, as the
// container toolkit does with the real one.
func stubNvidiaSMI(t *testing.T, script string) {
	t.Helper()
	bin := t.TempDir()
	require.NoError(t, os.WriteFile(filepath.Join(bin, "nvidia-smi"), []byte("#!/bin/sh\n"+script+"\n"), 0700)) // #nosec G306 -- Executable test fixture in a private temporary directory.
	t.Setenv("PATH", bin+":"+os.Getenv("PATH"))
}

func TestReadSMIXML(t *testing.T) {
	fixture, err := filepath.Abs(filepath.Join("testdata", "smi", "licensed-multiple.xml"))
	require.NoError(t, err)
	for name, tc := range map[string]struct {
		script    string
		wantGPUs  int
		wantError string
	}{
		"captured output": {script: "cat " + fixture, wantGPUs: 2},
		// A warning on stdout, such as one about the infoROM, does not fail the read.
		"text around the XML": {script: "echo 'WARNING: infoROM is corrupted'; cat " + fixture + "; echo 'trailing text'", wantGPUs: 2},
		"output is not XML":   {script: "echo not xml", wantError: "parse nvidia-smi XML"},
	} {
		t.Run(name, func(t *testing.T) {
			stubNvidiaSMI(t, tc.script)
			out, err := readSMIXML(t.Context())
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
				return
			}
			require.NoError(t, err)
			require.Len(t, out.GPUs, tc.wantGPUs)
		})
	}
}

// The data is stdout only. A failure reports stderr, and a deadline or a
// cancellation is reported as such, not as the signal that stopped nvidia-smi.
func TestReadSMIXMLCommandOutcome(t *testing.T) {
	for name, tc := range map[string]struct {
		script    string
		timeout   time.Duration
		cancelled bool
		wantUUID  string
		wantError string
		wantCause error
	}{
		// A warning on stderr in the middle of the UUID would corrupt it if stderr were
		// part of the data.
		"stderr stays out of the data": {script: "printf '<nvidia_smi_log><gpu><uuid>GPU-'; echo 'a warning' >&2; printf '1</uuid></gpu></nvidia_smi_log>'", wantUUID: "GPU-1"},
		"failure reports stderr":       {script: "echo 'NVML mismatch' >&2; exit 1", wantError: "NVML mismatch"},
		"deadline":                     {script: "exec sleep 30", timeout: 100 * time.Millisecond, wantError: "nvidia-smi did not finish", wantCause: context.DeadlineExceeded},
		"cancelled":                    {script: "exec sleep 30", cancelled: true, wantCause: context.Canceled},
	} {
		t.Run(name, func(t *testing.T) {
			ctx := t.Context()
			if tc.timeout > 0 {
				var cancel context.CancelFunc
				ctx, cancel = context.WithTimeout(ctx, tc.timeout)
				defer cancel()
			}
			if tc.cancelled {
				var cancel context.CancelFunc
				ctx, cancel = context.WithCancel(ctx)
				cancel()
			}
			stubNvidiaSMI(t, tc.script)
			out, err := readSMIXML(ctx)
			if tc.wantCause != nil {
				require.ErrorIs(t, err, tc.wantCause)
			}
			if tc.wantError != "" {
				require.ErrorContains(t, err, tc.wantError)
			}
			if tc.wantCause != nil || tc.wantError != "" {
				require.Zero(t, out)
				return
			}
			require.NoError(t, err)
			require.Len(t, out.GPUs, 1)
			require.Equal(t, tc.wantUUID, out.GPUs[0].UUID)
		})
	}
}

// A child of nvidia-smi that keeps stdout open does not hold up the read.
func TestReadSMIXMLChildHoldsOutputOpen(t *testing.T) {
	stubNvidiaSMI(t, "sleep 5 & printf '<nvidia_smi_log/>'")
	start := time.Now()
	_, err := readSMIXML(t.Context())
	require.ErrorIs(t, err, exec.ErrWaitDelay)
	require.Less(t, time.Since(start), 4*time.Second)
}

// Without the nvidia-smi that the container toolkit injects, the read reports it
// missing.
func TestReadSMIXMLWithoutNvidiaSMI(t *testing.T) {
	t.Setenv("PATH", t.TempDir())
	_, err := readSMIXML(t.Context())
	require.ErrorIs(t, err, exec.ErrNotFound)
}
