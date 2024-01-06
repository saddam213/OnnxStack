namespace OnnxStack.StableDiffusion.Guards;

using Common;
using Diffusers;
using Enums;

public static class PipelineDiffuserGuard
{
    public static void AgainstDiffuserNotFoundOrSupportedException(IPipeline pipeline, DiffuserType diffuserType)
    {
        IDiffuser diffuser = pipeline.GetDiffuser(diffuserType);
        if (diffuser is null)
        {
            throw new DiffuserNotFoundOrSupportedException();
        }
    }
}