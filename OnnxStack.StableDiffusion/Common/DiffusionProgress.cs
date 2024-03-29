﻿using Microsoft.ML.OnnxRuntime.Tensors;

namespace OnnxStack.StableDiffusion.Common
{
    public record DiffusionProgress(string Message = default)
    {
        public int BatchMax { get; set; }
        public int BatchValue { get; set; }
        public DenseTensor<float> BatchTensor { get; set; }

        public int StepMax { get; set; }
        public int StepValue { get; set; }
        public DenseTensor<float> StepTensor { get; set; }
    }
}
