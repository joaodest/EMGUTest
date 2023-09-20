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
    const string imgPath = "C:\\Users\\joaod\\source\\repos\\One Week PK\\EMGUTest\\EMGUTest\\Images\\";
    public static void Main(string[] args)
    {
        String win = "Windows screen";

        Mat image = new Mat();


        List<String> images = new List<String>();

        images.Add("teste2.png");
        images.Add("img_test1.png");
        images.Add("teste3.png");


        image = CvInvoke.Imread($"{imgPath + images[2]}", ImreadModes.Color);


        var img = ShapeDetection.ProcessImage(image);

        CvInvoke.Imshow("arquivo.jpg", img);
        CvInvoke.WaitKey(0);
        CvInvoke.DestroyWindow(win);

    }

}




