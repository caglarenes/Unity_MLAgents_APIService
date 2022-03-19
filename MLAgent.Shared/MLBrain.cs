using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Reflection;

namespace MLAgent.Shared
{
    public class MLBrain
    {
        Stopwatch st = new Stopwatch();
        ITransformer? onnxPredictionPipeline;
        PredictionEngine<InputClass, OutputClass>? onnxPredictionEngine;

        public MLBrain()
        {
            MLContext mlContext = new MLContext();
            onnxPredictionPipeline = GetPredictionPipeline(mlContext);
            onnxPredictionEngine = mlContext.Model.CreatePredictionEngine<InputClass, OutputClass>(onnxPredictionPipeline);
        }

        ITransformer GetPredictionPipeline(MLContext mlContext)
        {
            var inputColumns = new string[] { "vector_observation", "action_masks" };

            var outputColumns = new string[] { "discrete_actions", "action" };

            var modelLocation = $@"{Path.GetDirectoryName(Assembly.GetEntryAssembly().Location)}\SoccerTwos.onnx";

            var onnxPredictionPipeline = mlContext.Transforms.ApplyOnnxModel(outputColumnNames: outputColumns, inputColumnNames: inputColumns, modelLocation);

            var emptyDv = mlContext.Data.LoadFromEnumerable(new InputClass[] { });

            return onnxPredictionPipeline.Fit(emptyDv);
        }

        public OutputClass RequestPrediction(InputClass userInput)
        {
            st.Start();
            var prediction = onnxPredictionEngine.Predict(userInput);
            st.Stop();
            Console.WriteLine("Total Calculation Time in Ticks: " + st.Elapsed.Ticks);

            Console.WriteLine($"Action 1: {prediction.actions[0]}");
            Console.WriteLine($"Action 2: {prediction.actions[1]}");
            Console.WriteLine($"Action 3: {prediction.actions[2]}");
            Console.WriteLine($"Action 4: {prediction.actions[3]}");
            Console.WriteLine($"Action 5: {prediction.actions[4]}");
            Console.WriteLine($"Action 6: {prediction.actions[5]}");
            Console.WriteLine($"Action 7: {prediction.actions[6]}");
            Console.WriteLine($"Action 8: {prediction.actions[7]}");
            Console.WriteLine($"Action 9: {prediction.actions[8]}");

            Console.WriteLine($"Action Mask 1: {prediction.discreateActions[0]}");
            Console.WriteLine($"Action Mask 2: {prediction.discreateActions[1]}");
            Console.WriteLine($"Action Mask 3: {prediction.discreateActions[2]}");
            Console.WriteLine($"Action Mask 4: {prediction.discreateActions[3]}");
            Console.WriteLine($"Action Mask 5: {prediction.discreateActions[4]}");
            Console.WriteLine($"Action Mask 6: {prediction.discreateActions[5]}");
            Console.WriteLine($"Action Mask 7: {prediction.discreateActions[6]}");
            Console.WriteLine($"Action Mask 8: {prediction.discreateActions[7]}");
            Console.WriteLine($"Action Mask 9: {prediction.discreateActions[8]}");
            st.Reset();

            return prediction;
        }

        public class InputClass
        {
            [ColumnName("vector_observation"), VectorType(336)]
            public float[] floats { get; set; } = new float[336];
            [ColumnName("action_masks"), VectorType(9)]
            public float[] actionMasks { get; set; } = new float[9];

            public bool IsModelValid()
            {
                if (floats.Count() != 336)
                {
                    return false;
                }

                if (actionMasks.Count() != 9)
                {
                    return false;
                }

                return true;
            }
        }

        public class OutputClass
        {
            [ColumnName("discrete_actions"), VectorType(9)]
            public float[] discreateActions { get; set; } = new float[9];
            [ColumnName("action"), VectorType(9)]
            public float[] actions { get; set; } = new float[9];
        }
    }
}
