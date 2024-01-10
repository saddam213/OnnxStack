using Microsoft.ML.OnnxRuntime.Tensors;
using System;

namespace OnnxStack.Core
{
    public static class TensorExtension
    {
        /// <summary>
        /// Concatenates the specified tensors along the 0 axis.
        /// </summary>
        /// <param name="tensor1">The tensor1.</param>
        /// <param name="tensor2">The tensor2.</param>
        /// <param name="axis">The axis.</param>
        /// <returns></returns>
        /// <exception cref="System.NotImplementedException">Only axis 0 is supported</exception>
        public static DenseTensor<float> Concatenate(this DenseTensor<float> tensor1, DenseTensor<float> tensor2, int axis = 0)
        {
            if (tensor1 == null)
                return tensor2.ToDenseTensor();

            if (axis != 0 && axis != 2)
                throw new NotImplementedException("Only axis 0, 2 is supported");

            if (axis == 2)
                return Concatenate(tensor1, tensor2);

            var dimensions = tensor1.Dimensions.ToArray();
            dimensions[0] += tensor2.Dimensions[0];

            var buffer = new float[tensor1.Length + tensor2.Length].AsMemory();
            tensor1.Buffer.CopyTo(buffer[..(int)tensor1.Length]);
            tensor2.Buffer.CopyTo(buffer[(int)tensor1.Length..]);
            return new DenseTensor<float>(buffer, dimensions);
        }
    }
}
