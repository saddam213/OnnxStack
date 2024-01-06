namespace OnnxStack.Core.Video;

using System;

public class VideoTensorNotSupportedException : NotSupportedException
{
    public override string Message => "VideoTensor is not supported.";
}