namespace OnnxStack.Core.Exceptions;

using System;

public class InvalidPngHeaderException : Exception
{ 
    public override string Message => "Invalid PNG header.";
 }