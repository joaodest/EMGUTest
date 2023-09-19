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
        Mat image = new Mat("C:/Users/joaod/source/repos/One Week PK/EMGUTest/EMGUTest/assets/test1.png");

        ShapeDetection.ProcessImage(image);
         
    }


}

