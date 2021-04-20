WITH id_with_conv AS(
SELECT 
DISTINCT(client_id)
FROM `august-monument-187809.Realweb.4_attribution_table` 
WHERE session_with_conversion = True
),

path_table AS(
SELECT
client_id,
STRING_AGG(channel_group ORDER BY visit_start_time) AS path
FROM `august-monument-187809.Realweb.4_attribution_table` 
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