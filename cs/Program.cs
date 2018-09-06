using System;
using System.IO;
using TensorFlow;

namespace TFSTest
{
    class Program
    {
        static void Main(string[] args)
        {
            using(var graph = new TFGraph())
            {
                Console.WriteLine(TensorFlow.TFCore.Version);

                var exportdir = Path.Combine(Directory.GetCurrentDirectory(), "..", "exportdir");

                var session = new TFSession(graph).FromSavedModel(new TFSessionOptions(), null,
                    exportdir, new string[] {"serve"}, graph, new TFBuffer());
                //"serve" was picked up from saved_model_cli show --dir c:\dev\tensorflowsharp\exportdir

                Console.WriteLine(graph.ToString());

                var labelNames = new[] {"T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"};

                TestData.PrintImage();

                //get ready to run the prediction
                var runner = session.GetRunner();

                //sort out the input and output
                var input = graph["flatten_input"][0];
                var output = graph["dense_1/Softmax"][0];              
                runner.AddInput(input, new[] { TestData.TestClassification2D }); 
                runner.Fetch(output);

                //run the prediction
                var prediction = runner.Run();

                //now extract the results into an array and find the most likely one
                var outputArray = (float[,])prediction[0].GetValue(); 

                float maxPrediction = 0f;
                int maxIndex = 0;
                for(int i = 0; i < outputArray.Length; i++)
                {
                    if (outputArray[0, i] > maxPrediction)
                    {
                        maxPrediction = outputArray[0, i];
                        maxIndex = i;
                    }
                }

                // Show the resulting prediction
                Console.WriteLine();
                Console.WriteLine(labelNames[maxIndex]);

            }
        }
    }
}