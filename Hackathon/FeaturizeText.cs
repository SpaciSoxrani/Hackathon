using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;

namespace Hackathon
{
    public static class FeaturizeText
    {
        public static void Example()
        {

            // Create a new ML context, for ML.NET operations. It can be used for
            // exception tracking and logging, as well as the source of randomness.
            var mlContext = new MLContext();

            // Create a small dataset as an IEnumerable.
            var samples = new List<HousingData>()
            {
                new HousingData(){ Text = "sample number", CurrentPrice = 12000f },
                new HousingData(){ Text = "such as sentiment analysis, topic " +
                    "detection, intent identification etc." , CurrentPrice = 12000f},
                new HousingData(){ Text = "sample number", CurrentPrice = 12000f },
                new HousingData(){ Text = "such as sentiment analysis, topic " +
                    "detection, intent identification etc." , CurrentPrice = 12000f},
                new HousingData(){ Text = "sample number", CurrentPrice = 12000f },
                new HousingData(){ Text = "such as sentiment analysis, topic " +
                    "detection, intent identification etc." , CurrentPrice = 12000f},
                new HousingData(){ Text = "sample number", CurrentPrice = 12000f },
                new HousingData(){ Text = "such as sentiment analysis, topic " +
                    "detection, intent identification etc." , CurrentPrice = 12000f},
            };

            // Convert training data to IDataView.
            var dataview = mlContext.Data.LoadFromEnumerable(samples);

            // A pipeline for converting text into numeric features.
            // The following call to 'FeaturizeText' instantiates 
            // 'TextFeaturizingEstimator' with default parameters.
            // The default settings for the TextFeaturizingEstimator are
            //      * StopWordsRemover: None
            //      * CaseMode: Lowercase
            //      * OutputTokensColumnName: None
            //      * KeepDiacritics: false, KeepPunctuations: true, KeepNumbers:
            //          true
            //      * WordFeatureExtractor: NgramLength = 1
            //      * CharFeatureExtractor: NgramLength = 3, UseAllLengths = false
            // The length of the output feature vector depends on these settings.
            var textPipeline = mlContext.Transforms.Text.FeaturizeText("Features", new TextFeaturizingEstimator.Options()
            {
                OutputTokensColumnName = "OutputTokens",
                CaseMode = TextNormalizingEstimator.CaseMode.Lower,
                KeepNumbers = true,
                KeepPunctuations = false,
                WordFeatureExtractor = new WordBagEstimator.Options()
                {
                    MaximumNgramsCount = new int[] { 50 }
                },
                CharFeatureExtractor = null
            },
                "Text")
                .Append(mlContext.Transforms.NormalizeMinMax("Features"));

            // Fit to data.
            var textTransformer = textPipeline.Fit(dataview);

            IDataView transformedData = textTransformer.Transform(dataview);

            //// Create the prediction engine to get the features extracted from the
            //// text.
            //var predictionEngine = mlContext.Model.CreatePredictionEngine<TextData,
            //    TransformedTextData>(textTransformer);

            //// Convert the text into numeric features.
            //var prediction = predictionEngine.Predict(samples[0]);

            //// Print the length of the feature vector.
            //Console.WriteLine($"Number of Features: {prediction.Features.Length}");

            //// Print the first 10 feature values.
            //Console.Write("Features: ");
            //foreach (var f in prediction.Features)
            //    Console.Write($"{f:F4} ");

            // разделение данных на тренировочные и тестовые
            DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataview, testFraction: 0.2);
            IDataView trainData = dataSplit.TrainSet;
            IDataView testData = dataSplit.TestSet;




            // Define StochasticDualCoordinateAscent regression algorithm estimator
            var sdcaEstimator = mlContext.Regression.Trainers.Sdca();
            // Build machine learning model
            var trainedModel = sdcaEstimator.Fit(transformedData);



            var trainedModelParameters = trainedModel.Model as LinearRegressionModelParameters;
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public float[] Features { get; set; }
            public float Label { get; set; }
        }
        public class IssuePrediction
        {
            [ColumnName("PredictedLabel")]
            public string Area;
        }
    }
}