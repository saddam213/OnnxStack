namespace OnnxStack.Core.Exceptions;

using System;

public class NoVideoDataFoundException : ArgumentException
{
    public override string Message => "No video data found.";
}