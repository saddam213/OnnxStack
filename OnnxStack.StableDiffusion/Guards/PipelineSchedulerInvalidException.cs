namespace OnnxStack.StableDiffusion.Guards;

using System;
using OnnxStack.StableDiffusion.Enums;

public class PipelineSchedulerInvalidException : ArgumentException
{
    private SchedulerType _schedulerType;
    
    private DiffuserPipelineType _pipelineType;

    public override string Message { get; }
    
    public PipelineSchedulerInvalidException(DiffuserPipelineType pipelineType, SchedulerType schedulerType)
    {
        _schedulerType = schedulerType;
        _pipelineType = pipelineType;
        Message = $"Scheduler '{schedulerType}' is not compatible  with the `{pipelineType}` pipeline.";
    }
}