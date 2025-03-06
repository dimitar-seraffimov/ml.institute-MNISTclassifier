-- Create the predictions table to store MNIST prediction results
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    predicted_digit INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    true_label INTEGER,
    image_data BYTEA
);

-- Create an index on the timestamp column for faster queries
CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);

-- Create a view for digit-specific statistics
CREATE OR REPLACE VIEW digit_statistics AS
SELECT 
    predicted_digit,
    COUNT(*) as prediction_count,
    CAST(AVG(confidence) AS NUMERIC(10,4)) as avg_confidence
FROM predictions
GROUP BY predicted_digit
ORDER BY predicted_digit;

-- Create a view for accuracy statistics
CREATE OR REPLACE VIEW prediction_accuracy AS
SELECT
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN predicted_digit = true_label THEN 1 END) as correct_predictions,
    COUNT(CASE WHEN predicted_digit != true_label AND true_label IS NOT NULL THEN 1 END) as incorrect_predictions,
    COALESCE(
        CAST(
            100.0 * COUNT(CASE WHEN predicted_digit = true_label THEN 1 END) / 
            NULLIF(COUNT(CASE WHEN true_label IS NOT NULL THEN 1 END), 0)
        AS NUMERIC(10,2)), 
        0
    ) as accuracy_percentage,
    CAST(AVG(confidence) AS NUMERIC(10,4)) as avg_confidence
FROM predictions
WHERE true_label IS NOT NULL; 