namespace OnnxStack.StableDiffusion.Guards;

using System.Collections.Concurrent;
using System.Linq;
using OnnxStack.StableDiffusion.Common;
using OnnxStack.StableDiffusion.Enums;

public static class PipelineSchedulerGuard
{
    private static readonly ConcurrentDictionary<DiffuserPipelineType, SchedulerType[]>
        _pipelineSchedulerConfigs = new();

    private static bool _isPopulated => _pipelineSchedulerConfigs.Count > 0;

    // TODO: Use reflection and get this at app load via the IPipeline
    private static void PopulateAllowedSchedulersForPipelineTypes()
    {
        _pipelineSchedulerConfigs.Clear();
        
        // TODO: instantiate the DiffuserPipelineType via reflection so that we do not need to 'manually add' each pipeline type.
        _pipelineSchedulerConfigs.TryAdd(DiffuserPipelineType.InstaFlow,
            DiffuserPipelineType.InstaFlow.GetSchedulerTypes());
        _pipelineSchedulerConfigs.TryAdd(DiffuserPipelineType.LatentConsistency,
            DiffuserPipelineType.LatentConsistency.GetSchedulerTypes());
        _pipelineSchedulerConfigs.TryAdd(DiffuserPipelineType.LatentConsistencyXL,
            DiffuserPipelineType.LatentConsistencyXL.GetSchedulerTypes());
        _pipelineSchedulerConfigs.TryAdd(DiffuserPipelineType.StableDiffusion,
            DiffuserPipelineType.StableDiffusion.GetSchedulerTypes());

        _pipelineSchedulerConfigs.TryAdd(DiffuserPipelineType.StableDiffusionXL,
            DiffuserPipelineType.StableDiffusionXL.GetSchedulerTypes());
    }

    private static void LoadConfigDefinitions()
    {
        if (!_isPopulated)
        {
            PopulateAllowedSchedulersForPipelineTypes();
        }
    }


    public static SchedulerType AgainstInvalidSchedulerType(DiffuserPipelineType pipelineType,
        SchedulerType schedulerType)
    {
        LoadConfigDefinitions();
        bool isAllowedScheduler = _isAllowedScheduler(pipelineType, schedulerType);

        if (!isAllowedScheduler)
        {
            throw new PipelineSchedulerInvalidException(pipelineType, schedulerType);
        }

        return schedulerType;


        bool _isAllowedScheduler(DiffuserPipelineType pt, SchedulerType st)
        {
            return _pipelineSchedulerConfigs[pt].Contains(st);
        }
    }
}