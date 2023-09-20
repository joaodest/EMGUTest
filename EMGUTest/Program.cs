using System;
using System.Drawing;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Features2D;
using Emgu.CV.Reg;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using EMGUTest;

class Problem
{
    public static void Main(string[] args)
    {
        Mat image = new Mat("./assets/test1.png", CvInvoke.ImreadModes.Color);

        ShapeDetection.ProcessImage(image);
         
    }


}

