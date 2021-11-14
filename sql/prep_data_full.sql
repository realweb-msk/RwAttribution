-- QUERY TO PREPARE PATH DATA VIA SQL IN YOUR FAVOURITE DB
-- GOOGLE BIGQUERY STANDARD SQL

WITH id_with_conv AS(
SELECT 
DISTINCT(client_id)
-- PLEASE NOTE THAT attribution_table SHOULD HAVE FOLLOWING SCHEMA:
-- id (clientID, cookie, etc)
-- conversion_flag
-- touchpoint name(channel, sourceMedium, etc.)
-- order_col
-- FOR FURTHER DETAILS PLEASE VISIT https://github.com/realweb-msk/RwAttribution#Readme
FROM `projectID.datasetID.attribution_table`
WHERE session_with_conversion = True
),

path_table AS(
SELECT
client_id,
STRING_AGG(channel_group ORDER BY visit_start_time) AS path
FROM `projectID.datasetID.attribution_table`
WHERE client_id IN (SELECT * FROM id_with_conv)
GROUP BY client_id
ORDER BY client_id
),

total_table AS(
SELECT 
path, 
COUNT(client_id) AS conv
FROM path_table
GROUP BY path
ORDER BY conv desc
)

SELECT * 
FROM total_table
;