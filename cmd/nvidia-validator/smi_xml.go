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
	"encoding/xml"
	"errors"
	"fmt"
	"os/exec"
	"time"
)

// smiQueryTimeout bounds one run of nvidia-smi. A process stuck in the kernel, as
// on a wedged GPU, ignores the kill that the timeout sends, so the run then lasts
// until the process exits.
const smiQueryTimeout = 10 * time.Second

// smiXMLOutput is the part of the nvidia-smi -q -x document that this package
// reads.
type smiXMLOutput struct {
	XMLName xml.Name    `xml:"nvidia_smi_log"`
	GPUs    []smiXMLGPU `xml:"gpu"`
}

// smiXMLGPU is a gpu element of smiXMLOutput.
type smiXMLGPU struct {
	PCIBusID           string `xml:"id,attr"`
	UUID               string `xml:"uuid"`
	VirtualizationMode string `xml:"gpu_virtualization_mode>virtualization_mode"`
	// LicensedProducts is a slice so that a second element is visible.
	LicensedProducts []smiXMLLicensedProduct `xml:"vgpu_software_licensed_product"`
}

// smiXMLLicensedProduct is a vgpu_software_licensed_product element of smiXMLGPU.
type smiXMLLicensedProduct struct {
	Name   string `xml:"licensed_product_name"`
	Status string `xml:"license_status"`
}

// readSMIXML runs the nvidia-smi that the container toolkit injects and returns
// its parsed -q -x output. Text around the XML, such as a warning, is ignored.
func readSMIXML(ctx context.Context) (smiXMLOutput, error) {
	ctx, cancel := context.WithTimeout(ctx, smiQueryTimeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, "nvidia-smi", "-q", "-x")
	// Without a delay, Output keeps waiting for the output pipes after nvidia-smi
	// exits or is killed, for as long as any child process holds them open.
	cmd.WaitDelay = time.Second
	data, err := cmd.Output()
	if err != nil {
		if ctx.Err() != nil {
			return smiXMLOutput{}, fmt.Errorf("nvidia-smi did not finish within %s: %w", smiQueryTimeout, ctx.Err())
		}
		if exitErr, ok := errors.AsType[*exec.ExitError](err); ok {
			return smiXMLOutput{}, fmt.Errorf("%w; stderr: %q", err, exitErr.Stderr)
		}
		return smiXMLOutput{}, err
	}
	var out smiXMLOutput
	if err := xml.Unmarshal(data, &out); err != nil {
		return smiXMLOutput{}, fmt.Errorf("failed to parse nvidia-smi XML: %w", err)
	}
	return out, nil
}
