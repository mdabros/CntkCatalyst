using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Windows.Controls;
using System.Windows.Media.Imaging;

namespace CntkCatalyst.Examples
{
    public static class ImageUtilities
    {
        /// <summary>
        /// Originally from: https://github.com/anastasios-stamoulis/deep-learning-with-csharp-and-cntk/blob/master/DeepLearning/Util.cs
        /// </summary>
        static public BitmapImage BitmapToImageSource(Bitmap bitmap)
        {
            using (var memory = new MemoryStream())
            {
                bitmap.Save(memory, ImageFormat.Bmp);
                memory.Position = 0;

                var bitmapimage = new BitmapImage();
                bitmapimage.BeginInit();
                bitmapimage.StreamSource = memory;
                bitmapimage.CacheOption = BitmapCacheOption.OnLoad;
                bitmapimage.EndInit();

                return bitmapimage;
            }
        }

        /// <summary>
        /// Originally from: https://github.com/anastasios-stamoulis/deep-learning-with-csharp-and-cntk/blob/master/DeepLearning/Util.cs
        /// </summary>
        static public Bitmap CreateBitmap(float[] images, int gridIndex, int imageWidth, int imageHeight, 
            int imageChannels, bool adjustColorRange)
        {
            var pixelCountPrChannel = imageWidth * imageHeight;
            var maxValue = images.Skip(gridIndex * pixelCountPrChannel * imageChannels).Take(pixelCountPrChannel * imageChannels).Max();
            var minValue = images.Skip(gridIndex * pixelCountPrChannel * imageChannels).Take(pixelCountPrChannel * imageChannels).Min();

            var bitmap = new Bitmap(imageWidth, imageHeight);
            var sourceStart = gridIndex * pixelCountPrChannel;

            Func<int, int> getValue = (i) => (int)images[i];

            if (adjustColorRange)
            {
                getValue = (i) => (int)Rescale(images[i], minValue, maxValue);
            }

            for (int row = 0; row < imageHeight; row++)
            {
                for (int col = 0; col < imageWidth; col++)
                {
                    var pos = sourceStart + row * imageWidth + col;
                    var b = getValue(pos);
                    var g = (imageChannels == 1) ? b : getValue(pos + pixelCountPrChannel);
                    var r = (imageChannels == 1) ? b : getValue(pos + 2 * pixelCountPrChannel);
                    bitmap.SetPixel(col, row, Color.FromArgb(r, g, b));
                }
            }
            return bitmap;
        }

        static float Rescale(float value, float min, float max, float newMin = 0, float newMax = 255)
        {
            if (value == min)
            {
                value = newMin;
            }
            else if (value == max)
            {
                value = newMax;
            }
            else
            {
                value = newMin + (newMax - newMin) * (value - min) / (max - min);
            }

            return value;
        }
    }

    /// <summary>
    /// Originally from: https://github.com/anastasios-stamoulis/deep-learning-with-csharp-and-cntk/blob/master/DeepLearning/Util.cs
    /// </summary>
    public class PlotWindowBitMap : System.Windows.Window
    {
        public PlotWindowBitMap(string title, float[] images, int width, int height, int channels, 
            bool adjustColorRange)
        {
            var pixelCountPrChannel = width * height;
            var gridLength = (int)Math.Sqrt(images.Length / pixelCountPrChannel);
            var grid = new Grid();

            for (int row = 0; row < gridLength; row++)
            {
                grid.RowDefinitions.Add(new RowDefinition());
                for (int column = 0; column < gridLength; column++)
                {
                    if (row == 0)
                    {
                        grid.ColumnDefinitions.Add(new ColumnDefinition());
                    }

                    var gridIndex = (row * gridLength + column);
                    var bitmap = ImageUtilities.CreateBitmap(images, gridIndex, width, height, channels, 
                        adjustColorRange);

                    var image = new System.Windows.Controls.Image();
                    image.Source = ImageUtilities.BitmapToImageSource(bitmap);
                    image.Stretch = System.Windows.Media.Stretch.Fill;
                    grid.Children.Add(image);

                    Grid.SetRow(image, row);
                    Grid.SetColumn(image, column);
                }
            }

            Title = title;
            Content = grid;
            SizeToContent = System.Windows.SizeToContent.WidthAndHeight;
        }
    }
}
