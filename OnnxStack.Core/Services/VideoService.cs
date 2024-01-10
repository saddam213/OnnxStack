using FFMpegCore;
using Microsoft.ML.OnnxRuntime.Tensors;
using OnnxStack.Core.Config;
using OnnxStack.Core.Video;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace OnnxStack.Core.Services
{
    /// <summary>
    /// Service with basic handling of video for use in OnnxStack, Frame->Video and Video->Frames
    /// </summary>
    public class VideoService : IVideoService
    {
        private readonly OnnxStackConfig _configuration;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public VideoService(OnnxStackConfig configuration)
        {
            _configuration = configuration;
        }

        #region Public Members

        /// <summary>
        /// Gets the video information, Size, FPS, Duration etc.
        /// </summary>
        /// <param name="videoInput">The video input.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException">VideoTensor not supported</exception>
        /// <exception cref="ArgumentException">No video data found</exception>
        public async Task<VideoInfo> GetVideoInfoAsync(VideoInput videoInput, CancellationToken cancellationToken = default)
        {
            if (videoInput.VideoBytes is not null)
                return await GetVideoInfoAsync(videoInput.VideoBytes, cancellationToken);
            if (videoInput.VideoStream is not null)
                return await GetVideoInfoAsync(videoInput.VideoStream, cancellationToken);
            if (videoInput.VideoTensor is not null)
                throw new NotSupportedException("VideoTensor not supported");

            throw new ArgumentException("No video data found");
        }


        /// <summary>
        /// Gets the video information asynchronous.
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoInfo> GetVideoInfoAsync(Stream videoStream, CancellationToken cancellationToken = default)
        {
            using (var memoryStream = new MemoryStream())
            {
                await memoryStream.CopyToAsync(videoStream, cancellationToken);
                return await GetVideoInfoInternalAsync(memoryStream, cancellationToken);
            }
        }


        /// <summary>
        /// Gets the video information asynchronous.
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoInfo> GetVideoInfoAsync(byte[] videoBytes, CancellationToken cancellationToken = default)
        {
            using (var videoStream = new MemoryStream(videoBytes))
            {
                return await GetVideoInfoInternalAsync(videoStream, cancellationToken);
            }
        }


        /// <summary>
        /// Creates and MP4 video from a collection of PNG images.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoOutput> CreateVideoAsync(VideoFrames videoFrames, CancellationToken cancellationToken = default)
        {
            return await CreateVideoInternalAsync(videoFrames.Frames, videoFrames.Info.FPS, cancellationToken);
        }


        /// <summary>
        /// Creates and MP4 video from a collection of PNG images.
        /// </summary>
        /// <param name="videoTensor">The video tensor.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoOutput> CreateVideoAsync(DenseTensor<float> videoTensor, float videoFPS, CancellationToken cancellationToken = default)
        {
            var videoFrames = await videoTensor.ToVideoFramesAsBytesAsync().ToListAsync(cancellationToken);
            return await CreateVideoInternalAsync(videoFrames, videoFPS, cancellationToken);
        }


        /// <summary>
        /// Creates and MP4 video from a collection of PNG images.
        /// </summary>
        /// <param name="videoFrames">The video frames.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoOutput> CreateVideoAsync(IEnumerable<byte[]> videoFrames, float videoFPS, CancellationToken cancellationToken = default)
        {
            return await CreateVideoInternalAsync(videoFrames, videoFPS, cancellationToken);
        }


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoInput">The video input.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="NotSupportedException">VideoTensor not supported</exception>
        /// <exception cref="ArgumentException">No video data found</exception>
        public async Task<VideoFrames> CreateFramesAsync(VideoInput videoInput, float videoFPS, CancellationToken cancellationToken = default)
        {

            if (videoInput.VideoBytes is not null)
                return await CreateFramesAsync(videoInput.VideoBytes, videoFPS, cancellationToken);
            if (videoInput.VideoStream is not null)
                return await CreateFramesAsync(videoInput.VideoStream, videoFPS, cancellationToken);
            if (videoInput.VideoFrames is not null)
                return videoInput.VideoFrames;
            if (videoInput.VideoTensor is not null)
                throw new NotSupportedException("VideoTensor not supported");

            throw new ArgumentException("No video data found");
        }


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoFrames> CreateFramesAsync(byte[] videoBytes, float videoFPS, CancellationToken cancellationToken = default)
        {
            var videoInfo = await GetVideoInfoAsync(videoBytes, cancellationToken);
            var videoFrames = await CreateFramesInternalAsync(videoBytes, videoFPS, cancellationToken).ToListAsync(cancellationToken);
            videoInfo = videoInfo with { FPS = videoFPS };
            return new VideoFrames(videoInfo, videoFrames);
        }


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        /// <param name="videoFPS">The video FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async Task<VideoFrames> CreateFramesAsync(Stream videoStream, float videoFPS, CancellationToken cancellationToken = default)
        {
            using (var memoryStream = new MemoryStream())
            {
                await memoryStream.CopyToAsync(videoStream, cancellationToken).ConfigureAwait(false);
                var videoBytes = memoryStream.ToArray();
                var videoInfo = await GetVideoInfoAsync(videoBytes, cancellationToken);
                var videoFrames = await CreateFramesInternalAsync(videoBytes, videoFPS, cancellationToken).ToListAsync(cancellationToken);
                return new VideoFrames(videoInfo, videoFrames);
            }
        }


        /// <summary>
        /// Streams frames as PNG as they are processed from a video source
        /// </summary>
        /// <param name="videoBytes">The video bytes.</param>
        /// <param name="targetFPS">The target FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public IAsyncEnumerable<byte[]> StreamFramesAsync(byte[] videoBytes, float targetFPS, CancellationToken cancellationToken = default)
        {
            return CreateFramesInternalAsync(videoBytes, targetFPS, cancellationToken);
        }

        #endregion

        #region Private Members


        /// <summary>
        /// Gets the video information.
        /// </summary>
        /// <param name="videoStream">The video stream.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<VideoInfo> GetVideoInfoInternalAsync(MemoryStream videoStream, CancellationToken cancellationToken = default)
        {
            var result = await FFProbe.AnalyseAsync(videoStream).ConfigureAwait(false);
            return new VideoInfo(result.PrimaryVideoStream.Width, result.PrimaryVideoStream.Height, result.Duration, (int)result.PrimaryVideoStream.FrameRate);
        }


        /// <summary>
        /// Creates an MP4 video from a collection of PNG frames
        /// </summary>
        /// <param name="imageData">The image data.</param>
        /// <param name="fps">The FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async Task<VideoOutput> CreateVideoInternalAsync(IEnumerable<byte[]> imageData, float fps = 15, CancellationToken cancellationToken = default)
        {
            string tempVideoPath = GetTempFilename();
            try
            {
                // Analyze first fram to get some details
                var frameInfo = await GetVideoInfoAsync(imageData.First());
                var aspectRatio = (double)frameInfo.Width / frameInfo.Height;
                using (var videoWriter = CreateWriter(tempVideoPath, fps, aspectRatio))
                {
                    // Start FFMPEG
                    videoWriter.Start();
                    foreach (var image in imageData)
                    {
                        // Write each frame to the input stream of FFMPEG
                        await videoWriter.StandardInput.BaseStream.WriteAsync(image, cancellationToken);
                    }

                    // Done close stream and wait for app to process
                    videoWriter.StandardInput.BaseStream.Close();
                    await videoWriter.WaitForExitAsync(cancellationToken);

                    // Read result from temp file
                    var videoResult = await File.ReadAllBytesAsync(tempVideoPath, cancellationToken);

                    // Analyze the result
                    var videoInfo = await GetVideoInfoAsync(videoResult);
                    return new VideoOutput(videoResult, videoInfo);
                }
            }
            finally
            {
                DeleteTempFile(tempVideoPath);
            }
        }


        /// <summary>
        /// Creates a collection of PNG frames from a video source
        /// </summary>
        /// <param name="videoData">The video data.</param>
        /// <param name="fps">The FPS.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        /// <exception cref="Exception">Invalid PNG header</exception>
        private async IAsyncEnumerable<byte[]> CreateFramesInternalAsync(byte[] videoData, float fps = 15, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            string tempVideoPath = GetTempFilename();
            try
            {
                await File.WriteAllBytesAsync(tempVideoPath, videoData, cancellationToken);
                using (var ffmpegProcess = CreateReader(tempVideoPath, fps))
                {
                    // Start FFMPEG
                    ffmpegProcess.Start();

                    // FFMPEG output stream
                    var processOutputStream = ffmpegProcess.StandardOutput.BaseStream;

                    // Buffer to hold the current image
                    var buffer = new byte[20480000];

                    var currentIndex = 0;
                    while (!cancellationToken.IsCancellationRequested)
                    {
                        // Reset the index new PNG
                        currentIndex = 0;

                        // Read the PNG Header
                        if (await processOutputStream.ReadAsync(buffer, currentIndex, 8, cancellationToken) <= 0)
                            break;

                        currentIndex += 8;// header length

                        if (!IsImageHeader(buffer))
                            throw new Exception("Invalid PNG header");

                        // loop through each chunk
                        while (true)
                        {
                            // Read the chunk header
                            await processOutputStream.ReadAsync(buffer, currentIndex, 12, cancellationToken);

                            var chunkIndex = currentIndex;
                            currentIndex += 12; // Chunk header length

                            // Get the chunk's content size in bytes from the header we just read
                            var totalSize = buffer[chunkIndex] << 24 | buffer[chunkIndex + 1] << 16 | buffer[chunkIndex + 2] << 8 | buffer[chunkIndex + 3];
                            if (totalSize > 0)
                            {
                                var totalRead = 0;
                                while (totalRead < totalSize)
                                {
                                    int read = await processOutputStream.ReadAsync(buffer, currentIndex, totalSize - totalRead, cancellationToken);
                                    currentIndex += read;
                                    totalRead += read;
                                }
                                continue;
                            }

                            // If the size is 0 and is the end of the image
                            if (totalSize == 0 && IsImageEnd(buffer, chunkIndex))
                                break;
                        }

                        // Return Image stream
                        using (var imageStream = new MemoryStream(buffer, 0, currentIndex))
                            yield return imageStream.ToArray();
                    }

                    if (cancellationToken.IsCancellationRequested)
                        ffmpegProcess.Kill();
                }
            }
            finally
            {
                DeleteTempFile(tempVideoPath);
            }
        }


        /// <summary>
        /// Gets the temporary filename.
        /// </summary>
        /// <returns></returns>
        private string GetTempFilename()
        {
            if (!Directory.Exists(_configuration.TempPath))
                Directory.CreateDirectory(_configuration.TempPath);

            return Path.Combine(_configuration.TempPath, $"{Path.GetFileNameWithoutExtension(Path.GetRandomFileName())}.mp4");
        }


        /// <summary>
        /// Deletes the temporary file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        private void DeleteTempFile(string filename)
        {
            try
            {
                if (File.Exists(filename))
                    File.Delete(filename);
            }
            catch (Exception)
            {
                // File in use, Log
            }
        }


        /// <summary>
        /// Creates FFMPEG video reader process.
        /// </summary>
        /// <param name="inputFile">The input file.</param>
        /// <param name="fps">The FPS.</param>
        /// <returns></returns>
        private Process CreateReader(string inputFile, float fps)
        {
            var ffmpegProcess = new Process();
            ffmpegProcess.StartInfo.FileName = _configuration.FFmpegPath;
            ffmpegProcess.StartInfo.Arguments = $"-hide_banner -loglevel error -i \"{inputFile}\" -c:v png  -r {fps} -f image2pipe -";
            ffmpegProcess.StartInfo.RedirectStandardOutput = true;
            ffmpegProcess.StartInfo.UseShellExecute = false;
            ffmpegProcess.StartInfo.CreateNoWindow = true;
            return ffmpegProcess;
        }


        /// <summary>
        /// Creates FFMPEG video writer process.
        /// </summary>
        /// <param name="outputFile">The output file.</param>
        /// <param name="fps">The FPS.</param>
        /// <param name="aspectRatio">The aspect ratio.</param>
        /// <returns></returns>
        private Process CreateWriter(string outputFile, float fps, double aspectRatio)
        {
            var ffmpegProcess = new Process();
            ffmpegProcess.StartInfo.FileName = _configuration.FFmpegPath;
            ffmpegProcess.StartInfo.Arguments = $"-hide_banner -loglevel error -framerate {fps:F4} -i - -c:v libx264 -movflags +faststart -vf format=yuv420p -aspect {aspectRatio} {outputFile}";
            ffmpegProcess.StartInfo.RedirectStandardInput = true;
            ffmpegProcess.StartInfo.UseShellExecute = false;
            ffmpegProcess.StartInfo.CreateNoWindow = true;
            return ffmpegProcess;
        }


        /// <summary>
        /// Determines whether we are at the start of a PNG image in the specified buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The offset.</param>
        /// <returns>
        ///   <c>true</c> if the start of a PNG image sequence is detected<c>false</c>.
        /// </returns>
        private static bool IsImageHeader(byte[] buffer)
        {
            // PNG Header http://www.libpng.org/pub/png/spec/1.2/PNG-Structure.html#PNG-file-signature
            if (buffer[0] != 0x89
             || buffer[1] != 0x50
             || buffer[2] != 0x4E
             || buffer[3] != 0x47
             || buffer[4] != 0x0D
             || buffer[5] != 0x0A
             || buffer[6] != 0x1A
             || buffer[7] != 0x0A)
                return false;

            return true;
        }


        /// <summary>
        /// Determines whether we are at the end of a PNG image in the specified buffer.
        /// </summary>
        /// <param name="buffer">The buffer.</param>
        /// <param name="offset">The offset.</param>
        /// <returns>
        ///   <c>true</c> if the end of a PNG image sequence is detected<c>false</c>.
        /// </returns>
        private static bool IsImageEnd(byte[] buffer, int offset)
        {
            return buffer[offset + 4] == 0x49  // I
                && buffer[offset + 5] == 0x45  // E
                && buffer[offset + 6] == 0x4E  // N
                && buffer[offset + 7] == 0x44; // D
        }
    }

    #endregion
}
