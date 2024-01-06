namespace OnnxStack.StableDiffusion.Guards;

using System;

public class DiffuserNotFoundOrSupportedException : Exception
{
    public override string Message => "Diffuser not found or not supported.";


}