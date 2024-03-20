namespace OnnxStack.TextGeneration.Models
{ 
    public class SearchOptionsModel
    {
        public int TopK { get; set; } = 50;
        public float TopP { get; set; } = 0.95f;
        public float Temperature { get; set; } = 1;
        public float RepetitionPenalty { get; set; } = 0.9f;
        public bool PastPresentShareBuffer { get; set; } = false;
        public int NumReturnSequences { get; set; } = 1;
        public int NumBeams { get; set; } = 1;
        public int NoRepeatNgramSize { get; set; } = 0;
        public int MinLength { get; set; } = 0;
        public int MaxLength { get; set; } = 512;
        public float LengthPenalty { get; set; } = 1;
        public float DiversityPenalty { get; set; } = 0;
        public bool EarlyStopping { get; set; } = true;
        public bool DoSample { get; set; } = false;
    }
}