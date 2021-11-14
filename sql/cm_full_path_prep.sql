-- QUERY TO EXTRACT AND PREPARE DATA FROM CAMPAIGN MANAGER'S FULL PATH REPORT
-- https://support.google.com/campaignmanager/answer/10010860?hl=en
-- GOOGLE BIGQUERY STANDARD SQL

WITH path AS(
SELECT
path_pattern_id,
-- In our case field "ad" is one with information about touchpoint
STRING_AGG(IFNULL(ad, 'undefined'), '^' ORDER BY path_enven_index) AS path,
STRING_AGG(IFNULL(event_type, 'none'), '^' ORDER BY path_enven_index) AS path_events,
IF(event_type = 'FLOODLIGHT', 1, 0) AS path_with_conversion
FROM `projectID.datasetID.CM_full_path_table`
WHERE path_enven_index IS NOT NULL
GROUP BY path_pattern_id, event_type
),

conv AS(
SELECT
path_pattern_id,
SUM(total_paths) AS total_paths
FROM `projectID.datasetID.CM_full_path_table`
GROUP BY path_pattern_id
),

test AS(
SELECT *
FROM `projectID.datasetID.CM_full_path_table`
WHERE total_conversions > 0
)

SELECT
path.*,
conv.total_paths
FROM path
LEFT JOIN conv
ON path.path_pattern_id = conv.path_pattern_id
;