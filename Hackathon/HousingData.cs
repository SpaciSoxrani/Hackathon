using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Hackathon
{
    class HousingData
    {
        [LoadColumn(0)]
        public string Text { get; set; }

        [LoadColumn(1, 3)]
        [VectorType(3)]
        public float[] HistoricalPrices { get; set; }

        [LoadColumn(4)]
        [ColumnName("Label")]
        public float CurrentPrice { get; set; }
    }
}
