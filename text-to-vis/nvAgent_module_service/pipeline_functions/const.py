MAX_ROUND = 3  # max try times of one agent talk

processor_template = """
You are an experienced and professional database administrator. Given a database schema and a user query, your task is to analyze the query, filter the relevant schema, generate an optimized representation, and classify the query difficulty.

Now you can think step by step, following these instructions below.
【Instructions】
1. Schema Filtering:
    - Identify the tables and columns that are relevant to the user query.
    - Only exclude columns that are completely irrelevant.
    - The output should be {{tables: [columns]}}.
    - Keep the columns needed to be primary keys and foreign keys in the filtered schema.
    - Keep the columns that seem to be similar with other columns of another table.
    
2. New Schema Generation:
    - Generate a new schema of the filtered schema, based on the given database schema and your filtered schema.

3. Augmented Explanation:
    - Provide a concise summary of the filtered schema to give additional knowledge.
    - Include the number of tables, total columns, and any notable relationships or patterns.

4. Classification:
    - For the database new schema, classify it as SINGLE or MULTIPLE based on the tables number.
        if tables number >= 2: predict MULTIPLE
        elif only one table: predict SINGLE
    
==============================
Here is a typical example:
【Database Schema】
【DB_ID】 dorm_1
【Schema】
# Table: Student
[
  (stuid, And This is a id type column),
  (lname, Value examples: ['Smith', 'Pang', 'Lee', 'Adams', 'Nelson', 'Wilson'].),
  (fname, Value examples: ['Eric', 'Lisa', 'David', 'Sarah', 'Paul', 'Michael'].),
  (age, Value examples: [18, 20, 17, 19, 21, 22].),
  (sex, Value examples: ['M', 'F'].),
  (major, Value examples: [600, 520, 550, 50, 540, 100].),
  (advisor, And this is a number type column),
  (city code, Value examples: ['PIT', 'BAL', 'NYC', 'WAS', 'HKG', 'PHL'].)
]
# Table: Dorm
[
  (dormid, And This is a id type column),
  (dorm name, Value examples: ['Anonymous Donor Hall', 'Bud Jones Hall', 'Dorm-plex 2000', 'Fawlty Towers', 'Grad Student Asylum', 'Smith Hall'].),
  (student capacity, Value examples: [40, 85, 116, 128, 256, 355].),
  (gender, Value examples: ['X', 'F', 'M'].)
]
# Table: Dorm_amenity
[
  (amenid, And This is a id type column),
  (amenity name, Value examples: ['4 Walls', 'Air Conditioning', 'Allows Pets', 'Carpeted Rooms', 'Ethernet Ports', 'Heat'].)
]
# Table: Has_amenity
[
  (dormid, And This is a id type column),
  (amenid, And This is a id type column)
]
# Table: Lives_in
[
  (stuid, And This is a id type column),
  (dormid, And This is a id type column),
  (room number, And this is a number type column)
]

【Query】
Find the first name of students who are living in the Smith Hall, and count them by a pie chart

【Filtered Schema】
```json
{{
  "Student": ["stuid", "fname"],
  "Dorm": ["dormid", "dorm name"],
  "Lives_in": ["stuid", "dormid"]
}}
```

【New Schema】
# Table: Student
[
  (stuid, And This is a id type column),
  (fname, Value examples: ['Eric', 'Lisa', 'David', 'Sarah', 'Paul', 'Michael'].),
]
# Table: Dorm
[
  (dormid, And This is a id type column),
  (dorm name, Value examples: ['Anonymous Donor Hall', 'Bud Jones Hall', 'Dorm-plex 2000', 'Fawlty Towers', 'Grad Student Asylum', 'Smith Hall'].),
]
# Table: Lives_in
[
  (stuid, And This is a id type column),
  (dormid, And This is a id type column),
]
【Augmented Explanation】
The filtered schema consists of 3 tables (Student, Dorm, and Lives_in) with a total of 6 relevant columns. There is a many-to-one relationship between Student and Dorm through the Lives_in junction table. The query involves joining these three tables to find students living in a specific dorm (Smith Hall).

Key points:
1. The Lives_in table acts as a bridge between Student and Dorm, allowing for the association of students with their dorms.
2. The 'dorm name' column in the Dorm table is crucial for filtering the specific dorm (Smith Hall).
3. The 'fname' column from the Student table is required for the final output.

【Classification】
MULTIPLE

==============================
Here is a new question:

【DB_ID】{db_id}
【Database Schema】
{db_schema}

【Query】
{query}

Now give your answer following this format strictly without other explanation:

【Filtered Schema】

【New Schema】

【Augmented Explanation】

【Classification】

"""

multiple_template = """
Given a 【Database schema】 with 【Augmented Explanation】 and a 【Question】, generate a valid VQL (Visualization Query Language) sentence. VQL is similar to SQL but includes visualization components.

【Background】
VQL Structure:
Visualize [TYPE] SELECT [COLUMNS] FROM [TABLES] [JOIN] [WHERE] [GROUP BY] [ORDER BY] [BIN BY]

You can consider a VQL sentence as "VIS TYPE + SQL + BINNING"
You must consider which part in the sketch is necessary, which is unnecessary, and construct a specific sketch for the natural language query.

Key Components:
1. Visualization Type: bar, pie, line, scatter, stacked bar, grouped line, grouped scatter
2. SQL Components: SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY
3. Binning: BIN [COLUMN] BY [INTERVAL], [INTERVAL]: [YEAR, MONTH, DAY, WEEKDAY]

When generating VQL, we should always consider special rules and constraints:
【Special Rules】
a. For simple visualizations:
    - SELECT exactly TWO columns, X-axis and Y-axis(usually aggregate function)
b. For complex visualizations (STACKED BAR, GROUPED LINE, GROUPED SCATTER):
    - SELECT exactly THREE columns in this order!!!:
        1. X-axis
        2. Y-axis (aggregate function)
        3. Grouping column
c. When "COLORED BY" is mentioned in the question:
    - Use complex visualization type(STACKED BAR for bar charts, GROUPED LINE for line charts, GROUPED SCATTER for scatter charts)
    - Make the "COLORED BY" column the third SELECT column
    - Do NOT include "COLORED BY" in the final VQL     
d. Aggregate Functions:
    - Use COUNT for counting occurrences
    - Use SUM only for numeric columns
    - When in doubt, prefer COUNT over SUM
e. Time based questions:
    - Always use BIN BY clause at the end of VQL sentence
    - When you meet the questions including "year", "month", "day", "weekday"
    - Avoid using window function, just use BIN BY to deal with time base queries
【Constraints】
- In SELECT <column>, make sure there are at least two selected!!!
- In FROM <table> or JOIN <table>, do not include unnecessary table
- Use only table names and column names from the given database schema
- Enclose string literals in single quotes
- If [Value examples] of <column> has 'None' or None, use JOIN <table> or WHERE <column> is NOT NULL is better
- Ensure GROUP BY precedes ORDER BY for distinct values
- NEVER use window functions in SQL

Now we could think step by step:
1. First choose visualize type and binning, then construct a specific sketch for the natural language query
2. Second generate SQL components following the sketch.
3. Third add Visualize type and BINNING into the SQL components to generate final VQL

Here are some examples:
==========

【Database schema】
# Table: Orders, (orders)
[
  (order_id, order id, And this is a id type column),
  (customer_id, customer id, And this is a id type column),
  (order_date, order date, Value examples: ['2023-01-15', '2023-02-20', '2023-03-10'].),
  (total_amount, total amount, Value examples: [100.00, 200.00, 300.00, 400.00, 500.00].)
]
# Table: Customers, (customers)
[
  (customer_id, customer id, And this is a id type column),
  (customer_name, customer name, Value examples: ['John', 'Emma', 'Michael', 'Sophia', 'William'].),
  (customer_type, customer type, Value examples: ['Regular', 'VIP', 'New'].)
]
【Augmented Explanation】
The filtered schema consists of 2 tables (Orders and Customers) with a total of 7 relevant columns. There is a one-to-many relationship between Customers and Orders through the customer_id foreign key.

Key points:
1. The Orders table contains information about individual orders, including the order date and total amount.
2. The Customers table contains customer information, including their name and type (Regular, VIP, or New).
3. The customer_id column links the two tables, allowing us to associate orders with specific customers.
4. The order_date column in the Orders table will be used for monthly grouping and binning.
5. The total_amount column in the Orders table needs to be summed for each group.
6. The customer_type column in the Customers table will be used for further grouping and as the third dimension in the stacked bar chart.

The query involves joining these two tables to analyze order amounts by customer type and month, which requires aggregation and time-based binning.

【Question】
Show the total order amount for each customer type by month in a stacked bar chart.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: STACKED BAR, BINNING: True
VQL Sketch:
Visualize STACKED BAR SELECT _ , _ , _ FROM _ JOIN _ ON _ GROUP BY _ BIN _ BY MONTH

Sub task 2: Second generate SQL components following the sketch.
Let's think step by step:
1. We need to select 3 columns for STACKED BAR chart, order_date as X-axis, SUM(total_amout) as Y-axis, customer_type as group column.
2. We need to join the Orders and Customers tables.
3. We need to group by customer type.
4. We do not need to use month function for MONTH.

```sql
SELECT O.order_date, SUM(O.total_amount), C.customer_type
FROM Orders AS O
JOIN Customers AS C ON O.customer_id = C.customer_id
GROUP BY C.customer_type
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize STACKED BAR SELECT O.order_date, SUM(O.total_amount), C.customer_type FROM Orders O JOIN Customers C ON O.customer_id = C.customer_id GROUP BY C.customer_type BIN O.order_date BY MONTH

==========

【Database Schema】
# Table: department
[
  (dept_name, dept name, Value examples: ['Accounting', 'Astronomy', 'Psychology', 'Pol. Sci.', 'Physics', 'Mech. Eng.'].),
]
# Table: course
[
  (course_id, course id, And this is a id type column),
  (dept_name, dept name, Value examples: ['Cybernetics', 'Finance', 'Psychology', 'Accounting', 'Mech. Eng.', 'Physics'].),
]
# Table: section
[
  (course_id, course id, And this is a id type column),
  (semester, semester, Value examples: ['Fall', 'Spring'].),
  (year, year, Value examples: [2002, 2006, 2003, 2007, 2010, 2008].),
]

【Augmented Explanation】
The filtered schema includes 3 tables (department, course, section) with a total of 5 relevant columns. The department table contains information about different departments, the course table provides details about courses including the department they belong to, and the section table holds data on course sections offered in different semesters and years. The relationship between department and course is one-to-many, as a department can offer multiple courses.

Key points:
1. The 'dept_name' column in the department table is essential for filtering courses offered by the Psychology department.
2. The 'course_id' column in the course table is needed to uniquely identify courses.
3. The 'semester' and 'year' columns in the section table are crucial for determining the number of courses offered by the Psychology department in each year.

【Question】
Find the number of courses offered by Psychology department in each year with a line chart.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: LINE, BINNING: True
VQL Sketch:
Visualize LINE SELECT _ , _ FROM _ JOIN _ ON _ WHERE _ BIN _ BY YEAR

Sub task 2: Second generate SQL components following the sketch.
Let's think step by step:
1. We need to select 2 columns for LINE chart, yeas as X-axis, COUNT(year) as Y-axis.
2. We need to join the course and section tables to get the number of courses offered by the Psychology department in each year.
3. We need to filter the courses by the Psychology department.
4. We do not need to use year function for YEAR.

```sql
SELECT S.year, COUNT(S.year)
FROM course AS C
JOIN section AS S ON C.course_id = S.course_id
WHERE C.dept_name = 'Psychology'
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize LINE SELECT S.year, COUNT(S.year) FROM course C JOIN section S ON C.course_id = S.course_id WHERE C.dept_name = 'Psychology' BIN S.year BY YEAR

==========

【Database Schema】
# Table: Products
[
  (Product_ID, product id, And this is a id type column),
  (Product_Name, product name, Value examples: ['Dinning', 'TV Equipments', 'Men suits', 'Special Dinning', 'Business Policy', 'Business Policy B'].),
]
# Table: Products_in_Events
[
  (Product_ID, product id, And this is a id type column),
  (Event_ID, event id, And this is a id type column),
]

【Augmented Explanation】
The filtered schema includes 2 tables (Products and Products_in_Events) with a total of 3 relevant columns. The Products table contains information about the products, while the Products_in_Events table links products to events they are associated with. The query aims to find products that are not part of any event.

Key points:
1. The 'Product_Name' column from the Products table is essential for identifying the names of products.
2. The relationship between Products and Products_in_Events allows for determining which products are not linked to any event.

【Question】
A bar chart for listing the number of the names of products that are not in any event, sort the number of product name in ascending order.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: BAR, BINNING: True
VQL Sketch:
Visualize BAR SELECT _ , _ FROM _ WHERE _ NOT IN (subquery) GROUP BY _ ORDER BY _

Sub task 2: Second generate SQL components following the sketch.
Let's think step by step:
1. We need to select 2 columns for BAR chart, Product_Name as X-axis, COUNT(Product_Name) as Y-axis
2. We need to find the Product_ID in the Products_in_Events.
3. We need to group by Product_Name
4. We need to sort by Y-axis, COUNT(Product_Name) in ascending order

Let's solve the subquery first:
```sql
SELECT Product_ID 
FROM Products_in_Events
```
Now, let's use this subquery in the main query:
```sql
SELECT Product_Name , COUNT(Product_Name) 
FROM Products 
WHERE Product_ID NOT IN 
(SELECT Product_ID FROM Products_in_Events) 
GROUP BY Product_Name 
ORDER BY COUNT(Product_Name) ASC
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize BAR SELECT Product_Name , COUNT(Product_Name) FROM Products WHERE Product_ID NOT IN (SELECT Product_ID FROM Products_in_Events) GROUP BY Product_Name ORDER BY COUNT(Product_Name) ASC

==========

【Database Schema】
# Table: Student
[
  (major, major, Value examples: [600, 520, 550, 50, 540, 100].),
  (Age, age, Value examples: [18, 20, 17, 19, 21, 22].),
]

【Augmented Explanation】
The filtered schema consists of the Student table with 2 relevant columns. The query focuses on the relationship between the maximum and minimum age for different majors, indicating a need to analyze the age distribution within each major category.

Key points:
1. The 'major' column in the Student table is essential for grouping students by their majors.
2. The 'Age' column provides the data for calculating the maximum and minimum age within each major group.

【Question】
Show the relationship between maximum age and minimum age for different majors in a group scatter chart.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: GROUPED SCATTER, BINNING: False
VQL Sketch:
Visualize GROUPED SCATTER SELECT _ , _ , _ FROM _ GROUP BY _

Sub task 2: Second generate SQL components following the sketch
Let's think step by step:
1. We need to select 3 columns for GROUPED SCATTER chart, MIN(Age) as X-axis, MAX(Age) as Y-axis, major as grouping column.
2. We need to group by major.

```sql
SELECT MIN(Age), MAX(Age), major
FROM Student
GROUP BY major
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize GROUPED SCATTER SELECT MIN(Age), MAX(Age), major FROM Student GROUP BY major

==========
NOW here is a new question:

【Database Schema】
{desc_str}

【Augmented Explanation】
{augmented_explanation}

【Question】
{query}

Now, please generate a VQL sentence for the database schema and question after thinking step by step.
(stop answering after you give the final vql)
"""

single_template = """
Given a 【Database schema】 and a 【Question】, generate a valid VQL (Visualization Query Language) sentence. VQL is similar to SQL but includes visualization components.

【Background】
VQL Structure:
Visualize [TYPE] SELECT [COLUMNS] FROM [TABLES] [JOIN] [WHERE] [GROUP BY] [ORDER BY] [BIN BY]

You can consider a VQL sentence as "VIS TYPE + SQL + BINNING"
You must consider which part in the sketch is necessary, which is unnecessary, and construct a specific sketch for the natural language query.

Key Components:
1. Visualization Type: bar, pie, line, scatter, stacked bar, grouped line, grouped scatter
2. SQL Components: SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY
3. Binning: BIN [COLUMN] BY [INTERVAL], [INTERVAL]: [YEAR, MONTH, DAY, WEEKDAY]

When generating VQL, we should always consider special rules and constraints:
【Special Rules】
a. For simple visualizations:
    - SELECT exactly TWO columns, X-axis and Y-axis(usually aggregate function)
b. For complex visualizations (STACKED BAR, GROUPED LINE, GROUPED SCATTER):
    - SELECT exactly THREE columns in this order!!!:
        1. X-axis
        2. Y-axis (aggregate function)
        3. Grouping column
c. When "COLORED BY" is mentioned in the question:
    - Use complex visualization type(STACKED BAR for bar charts, GROUPED LINE for line charts, GROUPED SCATTER for scatter charts)
    - Make the "COLORED BY" column the third SELECT column
    - Do NOT include "COLORED BY" in the final VQL     
d. Aggregate Functions:
    - Use COUNT for counting occurrences
    - Use SUM only for numeric columns
    - When in doubt, prefer COUNT over SUM
e. Time based questions:
    - Always use BIN BY clause at the end of VQL sentence
    - When you meet the questions including "year", "month", "day", "weekday"
    - Avoid using window function, just use BIN BY to deal with time base queries
【Constraints】
- In SELECT <column>, make sure there are at least two selected!!!
- In FROM <table> or JOIN <table>, do not include unnecessary table
- Use only table names and column names from the given database schema
- Enclose string literals in single quotes
- If [Value examples] of <column> has 'None' or None, use JOIN <table> or WHERE <column> is NOT NULL is better
- Ensure GROUP BY precedes ORDER BY for distinct values
- NEVER use window functions in SQL

Now we could think step by step:
1. First choose visualize type and binning, then construct a specific sketch for the natural language query
2. Second generate SQL components following the sketch.
3. Third add Visualize type and BINNING into the SQL components to generate final VQL

Here are some examples:
==========

【Database schema】
# Table: Orders, (orders)
[
  (order_id, order id, And this is a id type column),
  (customer_id, customer id, And this is a id type column),
  (order_date, order date, Value examples: ['2023-01-15', '2023-02-20', '2023-03-10'].),
  (total_amount, total amount, Value examples: [100.00, 200.00, 300.00, 400.00, 500.00].)
]
# Table: Customers, (customers)
[
  (customer_id, customer id, And this is a id type column),
  (customer_name, customer name, Value examples: ['John', 'Emma', 'Michael', 'Sophia', 'William'].),
  (customer_type, customer type, Value examples: ['Regular', 'VIP', 'New'].)
]

【Question】
Show the total order amount for each customer type by month in a stacked bar chart.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: STACKED BAR, BINNING: True
VQL Sketch:
Visualize STACKED BAR SELECT _ , _ , _ FROM _ JOIN _ ON _ GROUP BY _ BIN _ BY MONTH

Sub task 2: Second generate SQL components following the sketch.
Let's think step by step:
1. We need to select 3 columns for STACKED BAR chart, order_date as X-axis, SUM(total_amout) as Y-axis, customer_type as group column.
2. We need to join the Orders and Customers tables.
3. We need to group by customer type.
4. We do not need to use month function for MONTH.

```sql
SELECT O.order_date, SUM(O.total_amount), C.customer_type
FROM Orders AS O
JOIN Customers AS C ON O.customer_id = C.customer_id
GROUP BY C.customer_type
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize STACKED BAR SELECT O.order_date, SUM(O.total_amount), C.customer_type FROM Orders O JOIN Customers C ON O.customer_id = C.customer_id GROUP BY C.customer_type BIN O.order_date BY MONTH

==========

【Database Schema】
# Table: course, (course)
[
  (course_id, course id, Value examples: [101, 696, 656, 659]. And this is an id type column),
  (title, title, Value examples: ['Geology', 'Differential Geometry', 'Compiler Design', 'International Trade', 'Composition and Literature', 'Environmental Law'].),
  (dept_name, dept name, Value examples: ['Cybernetics', 'Finance', 'Psychology', 'Accounting', 'Mech. Eng.', 'Physics'].),
  (credits, credits, Value examples: [3, 4].)
]
# Table: section, (section)
[
  (course_id, course id, Value examples: [362, 105, 960, 468]. And this is an id type column),
  (sec_id, sec id, Value examples: [1, 2, 3]. And this is an id type column),
  (semester, semester, Value examples: ['Fall', 'Spring'].),
  (year, year, Value examples: [2002, 2006, 2003, 2007, 2010, 2008].),
  (building, building, Value examples: ['Saucon', 'Taylor', 'Lamberton', 'Power', 'Fairchild', 'Main'].),
  (room_number, room number, Value examples: [180, 183, 134, 143].),
  (time_slot_id, time slot id, Value examples: ['D', 'J', 'M', 'C', 'E', 'F']. And this is an id type column)
]

【Question】
Find the number of courses offered by Psychology department in each year with a line chart.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: LINE, BINNING: True
VQL Sketch:
Visualize LINE SELECT _ , _ FROM _ JOIN _ ON _ WHERE _ BIN _ BY YEAR

Sub task 2: Second generate SQL components following the sketch.
Let's think step by step:
1. We need to select 2 columns for LINE chart, yeas as X-axis, COUNT(year) as Y-axis.
2. We need to join the course and section tables to get the number of courses offered by the Psychology department in each year.
3. We need to filter the courses by the Psychology department.
4. We do not need to use year function for YEAR.

```sql
SELECT S.year, COUNT(S.year)
FROM course AS C
JOIN section AS S ON C.course_id = S.course_id
WHERE C.dept_name = 'Psychology'
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize LINE SELECT S.year, COUNT(S.year) FROM course C JOIN section S ON C.course_id = S.course_id WHERE C.dept_name = 'Psychology' BIN S.year BY YEAR

==========

【Database Schema】
# Table: Products, (products)
[
  (Product_ID, product id, Value examples: [1, 3, 5, 6]. And this is an id type column),
  (Product_Type_Code, product type code, Value examples: ['Food', 'Books', 'Electronics', 'Clothes'].),
  (Product_Name, product name, Value examples: ['Dinning', 'TV Equipments', 'Men suits', 'Special Dinning', 'Business Policy', 'Business Policy B'].),
  (Product_Price, product price, Value examples: [502.15, 932.25, 970.77, 1336.26].)
]
# Table: Products_in_Events, (products in events)
[
  (Product_ID, product id, And this is a id type column),
  (Event_ID, event id, And this is a id type column),
]

【Question】
A bar chart for listing the number of the names of products that are not in any event, sort the number of product name in ascending order.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: BAR, BINNING: True
VQL Sketch:
Visualize BAR SELECT _ , _ FROM _ WHERE _ NOT IN (subquery) GROUP BY _ ORDER BY _

Sub task 2: Second generate SQL components following the sketch.
Let's think step by step:
1. We need to select 2 columns for BAR chart, Product_Name as X-axis, COUNT(Product_Name) as Y-axis
2. We need to find the Product_ID in the Products_in_Events.
3. We need to group by Product_Name
4. We need to sort by Y-axis, COUNT(Product_Name) in ascending order

Let's solve the subquery first:
```sql
SELECT Product_ID 
FROM Products_in_Events
```
Now, let's use this subquery in the main query:
```sql
SELECT Product_Name , COUNT(Product_Name) 
FROM Products 
WHERE Product_ID NOT IN 
(SELECT Product_ID FROM Products_in_Events) 
GROUP BY Product_Name 
ORDER BY COUNT(Product_Name) ASC
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize BAR SELECT Product_Name , COUNT(Product_Name) FROM Products WHERE Product_ID NOT IN (SELECT Product_ID FROM Products_in_Events) GROUP BY Product_Name ORDER BY COUNT(Product_Name) ASC

==========

【Database Schema】
# Table: Student, (student)
[
  (StuID, stuid, Value examples: [1001, 1027, 1021, 1022]. And this is an id type column),
  (LName, lname, Value examples: ['Smith', 'Pang', 'Lee', 'Adams', 'Nelson', 'Wilson'].),
  (Fname, fname, Value examples: ['Eric', 'Lisa', 'David', 'Sarah', 'Paul', 'Michael'].),
  (Age, age, Value examples: [18, 20, 17, 19, 21, 22].),
  (Sex, sex, Value examples: ['M', 'F'].),
  (Major, major, Value examples: [600, 520, 550, 50, 540, 100].),
  (Advisor, advisor, Value examples: [2192, 1121, 8722, 2311].),
  (city_code, city code, Value examples: ['PIT', 'BAL', 'NYC', 'WAS', 'HKG', 'PHL'].)
]

【Question】
Show the relationship between maximum age and minimum age for different majors in a group scatter chart.

Decompose the task into sub tasks, considering 【Background】【Special Rules】【Constraints】, and generate the VQL after thinking step by step:

Sub task 1: First choose visualize type and binning, then construct a specific sketch for the natural language query
Visualize type: GROUPED SCATTER, BINNING: False
VQL Sketch:
Visualize GROUPED SCATTER SELECT _ , _ , _ FROM _ GROUP BY _

Sub task 2: Second generate SQL components following the sketch
Let's think step by step:
1. We need to select 3 columns for GROUPED SCATTER chart, MIN(Age) as X-axis, MAX(Age) as Y-axis, major as grouping column.
2. We need to group by major.

```sql
SELECT MIN(Age), MAX(Age), major
FROM Student
GROUP BY major
```

Sub task 3: Third add Visualize type and BINNING into the SQL components to generate final VQL
Final VQL:
Visualize GROUPED SCATTER SELECT MIN(Age), MAX(Age), major FROM Student GROUP BY major

==========
NOW here is a new question:

【Database Schema】
{desc_str}

【Question】
{query}

Now, please generate a VQL sentence for the database schema and question after thinking step by step.
(stop answering after you give the final vql)
"""

refiner_vql_template = """
As an AI assistant specializing in data visualization and VQL (Visualization Query Language), your task is to refine a VQL query that has resulted in an error. Please approach this task systematically, thinking step by step.
【Background】
VQL Structure:
Visualize [TYPE] SELECT [COLUMNS] FROM [TABLES] [JOIN] [WHERE] [GROUP BY] [ORDER BY] [BIN BY]

You can consider a VQL sentence as "VIS TYPE + SQL + BINNING"

Key Components:
1. Visualization Type: bar, pie, line, scatter, stacked bar, grouped line, grouped scatter
2. SQL Components: SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY
3. Binning: BIN [COLUMN] BY [INTERVAL], [INTERVAL]: [YEAR, MONTH, DAY, WEEKDAY]

When refining VQL, we should always consider special rules and constraints:
【Special Rules】
a. For simple visualizations:
    - SELECT exactly TWO columns, X-axis and Y-axis(usually aggregate function)
b. For complex visualizations (STACKED BAR, GROUPED LINE, GROUPED SCATTER):
    - SELECT exactly THREE columns in this order!!!:
        1. X-axis
        2. Y-axis (aggregate function)
        3. Grouping column
c. When "COLORED BY" is mentioned in the question:
    - Use complex visualization type(STACKED BAR for bar charts, GROUPED LINE for line charts, GROUPED SCATTER for scatter charts)
    - Make the "COLORED BY" column the third SELECT column
    - Do NOT include "COLORED BY" in the final VQL     
d. Aggregate Functions:
    - Use COUNT for counting occurrences
    - Use SUM only for numeric columns
    - When in doubt, prefer COUNT over SUM
e. Time based questions:
    - Always use BIN BY clause at the end of VQL sentence
    - When you meet the questions including "year", "month", "day", "weekday"
    - Avoid using time function, just use BIN BY to deal with time base queries
【Constraints】
- In FROM <table> or JOIN <table>, do not include unnecessary table
- Use only table names and column names from the given database schema
- Enclose string literals in single quotes
- If [Value examples] of <column> has 'None' or None, use JOIN <table> or WHERE <column> is NOT NULL is better
- ENSURE GROUP BY clause cannot contain aggregates
- NEVER use date functions in SQL

【Query】
{query}

【Database info】
{db_info}

【Current VQL】
{vql}

【Error】
{error}

Now, please analyze and refine the VQL, please provide:

【Explanation】
[Provide an brief explanation of your analysis process, the issues identified, and the changes made. Reference specific steps where relevant.]

【Corrected VQL】
[Present your corrected VQL here. Ensure it's on a single line without any line breaks.]

Remember:
- The SQL components must be parseable by DuckDB.
- Do not change rows when you generate the VQL.
- Always verify your answer carefully before submitting.
"""

refiner_python_template = """
As an AI assistant specializing in data visualization and Python, your task is to refine a Python code that has resulted in an error. Please approach this task systematically, thinking step by step.

【Query】
{query}

【Database info】
{db_info}

【Current Python Code】
{code}

【Error】
{error}

Now, please analyze and refine the Python code. Provide:

【Explanation】
[Provide a brief explanation of your analysis process, the issues identified, and the changes made. Reference specific steps where relevant.]

【Corrected Python Code】
```python
[Present your corrected Python code here. Ensure it's properly formatted and indented.]
```

Remember:
- The code must be executable in a Python environment with matplotlib or seaborn.
- Ensure the visualization matches the requirements of the original query.
- Always verify your answer carefully before submitting.
"""

basic_composer_template = """
Given a 【Database schema】 with 【Augmented Explanation】 and a 【Question】, generate a valid VQL (Visualization Query Language) sentence. VQL is similar to SQL but includes visualization components.

【Background】
VQL Structure:
Visualize [TYPE] SELECT [COLUMNS] FROM [TABLES] [JOIN] [WHERE] [GROUP BY] [ORDER BY] [BIN BY]

You can consider a VQL sentence as "VIS TYPE + SQL + BINNING"
You must consider which part in the sketch is necessary, which is unnecessary, and construct a specific sketch for the natural language query.

Key Components:
1. Visualization Type: bar, pie, line, scatter, stacked bar, grouped line, grouped scatter
2. SQL Components: SELECT, FROM, JOIN, WHERE, GROUP BY, ORDER BY
3. Binning: BIN [COLUMN] BY [INTERVAL], [INTERVAL]: [YEAR, MONTH, DAY, WEEKDAY]

When generating VQL, we should always consider special rules and constraints:
【Special Rules】
a. For simple visualizations:
    - SELECT exactly TWO columns, X-axis and Y-axis(usually aggregate function)
b. For complex visualizations (STACKED BAR, GROUPED LINE, GROUPED SCATTER):
    - SELECT exactly THREE columns in this order!!!:
        1. X-axis
        2. Y-axis (aggregate function)
        3. Grouping column
c. When "COLORED BY" is mentioned in the question:
    - Use complex visualization type(STACKED BAR for bar charts, GROUPED LINE for line charts, GROUPED SCATTER for scatter charts)
    - Make the "COLORED BY" column the third SELECT column
    - Do NOT include "COLORED BY" in the final VQL     
d. Aggregate Functions:
    - Use COUNT for counting occurrences
    - Use SUM only for numeric columns
    - When in doubt, prefer COUNT over SUM
e. Time based questions:
    - Always use BIN BY clause at the end of VQL sentence
    - When you meet the questions including "year", "month", "day", "weekday"
    - Avoid using window function, just use BIN BY to deal with time base queries
【Constraints】
- In SELECT <column>, make sure there are at least two selected!!!
- In FROM <table> or JOIN <table>, do not include unnecessary table
- Use only table names and column names from the given database schema
- Enclose string literals in single quotes
- If [Value examples] of <column> has 'None' or None, use JOIN <table> or WHERE <column> is NOT NULL is better
- Ensure GROUP BY precedes ORDER BY for distinct values
- NEVER use window functions in SQL

【Database Schema】
{desc_str}

【Augmented Explanation】
{augmented_explanation}

【Question】
{query}

Please generate a VQL sentence for the given database schema and question.
"""

without_filter_template = """
You are an experienced and professional database administrator. Given a database schema and a user query, your task is to analyze the query, filter the relevant schema, generate an optimized representation, and classify the query difficulty.

Now you can think step by step, following these instructions below.
【Instructions】
1. Augmented Explanation:
    - Provide a concise summary of the schema to give additional knowledge.
    - Include the number of tables, total columns, and any notable relationships or patterns.

2. Classification:
    - For the database new schema, classify it as SINGLE or MULTIPLE based on the tables number.
        if tables number >= 2: predict MULTIPLE
        elif only one table: predict SINGLE

==============================
Here is a new question:

【DB_ID】{db_id}
【Database Schema】
{db_schema}

【Query】
{query}

Now give your answer after thinking step by step:

【Augmented Explanation】

【Classification】

"""
