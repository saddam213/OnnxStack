using Microsoft.ML.OnnxRuntime;
using OnnxStack.Common.Config;
using OnnxStack.Core.Config;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace OnnxStack.UI.Models
{
    public class OnnxStackUIConfig : IConfigSection
    {
        public int DefaultDeviceId { get; set; }
        public int DefaultInterOpNumThreads { get; set; }
        public int DefaultIntraOpNumThreads { get; set; }
        public ExecutionMode DefaultExecutionMode { get; set; }
        public ExecutionProvider DefaultExecutionProvider { get; set; }
        public IEnumerable<ExecutionProvider> SupportedExecutionProviders => GetSupportedExecutionProviders();
        public ObservableCollection<UpscaleModelSetViewModel> UpscaleModelSets { get; set; } = new ObservableCollection<UpscaleModelSetViewModel>();
        public ObservableCollection<StableDiffusionModelSetViewModel> StableDiffusionModelSets { get; set; } = new ObservableCollection<StableDiffusionModelSetViewModel>();


        public IEnumerable<ExecutionProvider> GetSupportedExecutionProviders()
        {
#if DEBUG_DIRECTML || RELEASE_DIRECTML
            yield return ExecutionProvider.DirectML;
#elif DEBUG_CUDA || RELEASE_CUDA
            yield return ExecutionProvider.Cuda;
#elif DEBUG_TENSORRT || RELEASE_TENSORRT
            yield return ExecutionProvider.TensorRT;
#elif DEBUG_OPENVINO || RELEASE_OPENVINO
            yield return ExecutionProvider.OpenVino;
#endif
            yield return ExecutionProvider.Cpu;
        }

        public void Initialize()
        {
            DefaultExecutionProvider = SupportedExecutionProviders.Contains(DefaultExecutionProvider)
                ? DefaultExecutionProvider
                : SupportedExecutionProviders.First();
        }

    }
}
