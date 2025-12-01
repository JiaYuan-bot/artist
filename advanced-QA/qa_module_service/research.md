My research question is this:
In cloud database scenario, I have a cluster of compute nodes, each node will cache the data blocks retrived from remote cloud storage like S3. Then a query will be assigned to the node which already has the data that query needed. For example, query1 and query2 both access data item A, then, they will be assined to the same node, says node1. Usually this scheduling policy is achieved by consistent hashing. But if too much queries access data item A, node1 will be assigned too much queries, the load imbalanced problem arises among compute nodes.
Could you design a reinforcement learning policy to balance the data locality(queries access the same data need to be assinged to same compute node) and load balance problem.

Give me both detailed solution and code.

extend this function to a rpc service. The parameters should be a dict, not separate task,tentative_schema, execution_history.
I need protobuf file, rpc client, rpc server code example. Don't be too complex, I will modify them later.
def keyword_extraction(task: Any, tentative_schema: Dict[str, Any], execution_history: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extracts keywords from the task using an LLM chain call.

    Args:
        task (Any): The task object containing the evidence and question.
        tentative_schema (Dict[str, Any]): The current tentative schema.
        execution_history (Dict[str, Any]): The history of executions.

    Returns:
        Dict[str, Any]: A dictionary containing the extracted keywords.
    """
    request_kwargs = {
        "HINT": task.evidence,
        "QUESTION": task.question,
    }
    
    logging.info("Fetching prompt, engine, and parser from PipelineManager")
    prompt, engine, parser = PipelineManager().get_prompt_engine_parser()
    
    logging.info("Initiating asynchronous LLM chain call for keyword extraction")
    response = async_llm_chain_call(
        prompt=prompt, 
        engine=engine, 
        parser=parser,
        request_list=[request_kwargs],
        step="keyword_extraction",
        sampling_count=1
    )[0]
    
    keywords = response[0]
    result = {"keywords": keywords}
    
    logging.info(f"Keywords extracted: {keywords}")
    return result

I have several rpc function service, include keyword_extraction. I want to use call to call these functions, indicated by function_name. How can I achieve this? You can level params alone. I will take care of them

def call(self, function_name: str, node: Node,
             inputs: Dict[str, Any]) -> Dict[str, Any]:


├── database_utils
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── db_info.cpython-312.pyc
│   │   ├── execution.cpython-312.pyc
│   │   ├── schema.cpython-312.pyc
│   │   ├── schema_generator.cpython-312.pyc
│   │   └── sql_parser.cpython-312.pyc
│   ├── db_catalog
│   │   ├── __pycache__
│   │   │   ├── csv_utils.cpython-312.pyc
│   │   │   ├── preprocess.cpython-312.pyc
│   │   │   └── search.cpython-312.pyc
│   │   ├── csv_utils.py
│   │   ├── preprocess.py
│   │   └── search.py
│   ├── db_info.py
│   ├── db_values
│   │   ├── __pycache__
│   │   │   ├── preprocess.cpython-312.pyc
│   │   │   └── search.cpython-312.pyc
│   │   ├── preprocess.py
│   │   └── search.py
│   ├── execution.py
│   ├── schema.py
│   ├── schema_generator.py
│   └── sql_parser.py
├── env.example
├── idl
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── functions_pb2.cpython-312.pyc
│   │   └── functions_pb2_grpc.cpython-312.pyc
│   ├── gen_proto.sh
│   ├── proto
│   │   └── functions.proto
│   └── python
│       ├── __pycache__
│       │   ├── functions_pb2.cpython-312.pyc
│       │   └── functions_pb2_grpc.cpython-312.pyc
│       ├── functions_pb2.py
│       └── functions_pb2_grpc.py
├── llm
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── engine_configs.cpython-312.pyc
│   │   ├── models.cpython-312.pyc
│   │   ├── parsers.cpython-312.pyc
│   │   ├── prompts.cpython-312.pyc
│   │   └── utils.cpython-312.pyc
│   ├── engine_configs.py
│   ├── models.py
│   ├── parsers.py
│   ├── prompts.py
│   └── utils.py
├── main.py
├── pipeline_functions
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── keyword_extraction.cpython-312.pyc
│   │   └── utils.cpython-312.pyc
│   ├── candidate_generation.py
│   ├── column_filtering.py
│   ├── column_selection.py
│   ├── context_retrieval.py
│   ├── entity_retrieval.py
│   ├── evaluation.py
│   ├── keyword_extraction.py
│   ├── revision.py
│   ├── table_selection.py
│   └── utils.py
├── requirements.txt
├── research.md
├── runner
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-312.pyc
│   │   ├── database_manager.cpython-312.pyc
│   │   ├── logger.cpython-312.pyc
│   │   └── server.cpython-312.pyc
│   ├── database_manager.py
│   ├── logger.py
│   └── server.py
└── templates
    ├── template_candidate_generation.txt
    ├── template_column_filtering.txt
    ├── template_column_filtering_with_examples.txt
    ├── template_column_filtering_with_examples_llama.txt
    ├── template_column_selection.txt
    ├── template_finetuned_candidate_generation.txt
    ├── template_keyword_extraction.txt
    ├── template_revision.txt
    └── template_table_selection.tx

When I run python main.py
ModuleNotFoundError: No module named 'prompts'

  File "/data/nl2sql/butterfly/chess_function_service/llm/utils.py", line 4, in <module>
    from prompts import get_prompt
ModuleNotFoundError: No module named 'prompts'
from prompts import get_prompt
from models import get_llm_chain
from parsers import get_parser




INFO:workflow_controller.workflow_tracer:Node evaluation executed successfully in 0.012s, output: {'keys': {'task': {'question_id': 0, 'db_id': 'california_schools', 'question': 'What is the highest eligible free rate for K-12 students in the schools in Alameda County?', 'evidence': 'Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`', 'SQL': "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1", 'difficulty': 'simple'}, 'tentative_schema': {'frpm': ['CDSCode', 'Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name'], 'schools': ['CDSCode', 'County']}, 'execution_history': [{'node_type': 'keyword_extraction', 'keywords': ['highest eligible free rate', 'K-12 students', 'schools', 'Alameda County', 'Free Meal Count (K-12)', 'Enrollment (K-12)'], 'status': 'success'}, {'node_type': 'entity_retrieval', 'similar_columns': {'schools': ['School', 'County'], 'frpm': ['Free Meal Count (K-12)', 'Enrollment (K-12)']}, 'similar_values': {'frpm': {'School Name': ['Alameda County Community'], 'District Name': ['Alameda County Office of Education'], 'County Name': ['Alameda']}, 'schools': {'School': ['Alameda County Community'], 'District': ['Alameda Unified'], 'County': ['Alameda'], 'MailCity': ['Alameda'], 'City': ['Alameda'], 'GSserved': ['K-12'], 'GSoffered': ['K-12']}, 'satscores': {'dname': ['Alameda County Office of Education'], 'cname': ['Alameda'], 'sname': ['Avalon K-12']}}, 'status': 'success'}, {'node_type': 'context_retrieval', 'schema_with_descriptions': {'frpm': {'free meal count (k-12)': {'column_name': '', 'column_description': 'Free Meal Count (K-12)', 'value_description': 'eligible free rate = Free Meal Count / Enrollment'}, 'free meal count (ages 5-17)': {'column_name': '', 'column_description': 'Free Meal Count (Ages 5-17)', 'value_description': 'eligible free rate = Free Meal Count / Enrollment'}, 'frpm count (k-12)': {'column_name': '', 'column_description': 'Free or Reduced Price Meal Count (K-12)', 'value_description': 'eligible FRPM rate = FRPM / Enrollment'}}, 'schools': {'fundingtype': {'column_name': '', 'column_description': 'Indicates the charter school funding type', 'value_description': 'Values are as follows:  ·\xa0\xa0\xa0\xa0\xa0\xa0 Not in CS (California School) funding model  ·\xa0\xa0\xa0\xa0\xa0\xa0 Locally funded  ·\xa0\xa0\xa0\xa0\xa0\xa0 Directly funded'}, 'gsserved': {'column_name': 'grade span served.', 'column_description': 'It is the lowest grade and the highest grade of student enrollment as reported in the most recent certified CALPADS Fall 1 data collection. Only K–12 enrollment is reported through CALPADS. This field may differ from the grade span offered.', 'value_description': '1.\xa0\xa0\xa0\xa0 Only K–12 enrollment is reported through CALPADS  2.\xa0\xa0\xa0\xa0 Note: Special programs at independent study, alternative education, and special education schools will often exceed the typical grade span for schools of that type'}}}, 'status': 'success'}, {'node_type': 'column_filtering', 'tentative_schema': {'frpm': ['CDSCode', 'Academic Year', 'County Code', 'District Code', 'School Code', 'County Name', 'District Name', 'School Name', 'District Type', 'School Type', 'NSLP Provision Status', 'Charter School Number', 'IRC', 'High Grade', 'Enrollment (K-12)', 'Free Meal Count (K-12)', 'Percent (%) Eligible Free (K-12)', 'FRPM Count (K-12)', 'Percent (%) Eligible FRPM (K-12)', 'Enrollment (Ages 5-17)', 'Free Meal Count (Ages 5-17)', 'Percent (%) Eligible Free (Ages 5-17)', 'FRPM Count (Ages 5-17)', 'Percent (%) Eligible FRPM (Ages 5-17)'], 'satscores': ['cds', 'sname', 'dname', 'cname', 'enroll12'], 'schools': ['CDSCode', 'NCESDist', 'NCESSchool', 'StatusType', 'County', 'District', 'School', 'City', 'Zip', 'MailCity', 'MailZip', 'SOC', 'SOCType', 'EILCode', 'EILName', 'GSoffered', 'GSserved', 'Latitude']}, 'status': 'success'}, {'node_type': 'table_selection', 'tentative_schema': {'frpm': ['CDSCode', 'Academic Year', 'County Code', 'District Code', 'School Code', 'County Name', 'District Name', 'School Name', 'District Type', 'School Type', 'NSLP Provision Status', 'Charter School Number', 'IRC', 'High Grade', 'Enrollment (K-12)', 'Free Meal Count (K-12)', 'Percent (%) Eligible Free (K-12)', 'FRPM Count (K-12)', 'Percent (%) Eligible FRPM (K-12)', 'Enrollment (Ages 5-17)', 'Free Meal Count (Ages 5-17)', 'Percent (%) Eligible Free (Ages 5-17)', 'FRPM Count (Ages 5-17)', 'Percent (%) Eligible FRPM (Ages 5-17)'], 'schools': ['CDSCode', 'NCESDist', 'NCESSchool', 'StatusType', 'County', 'District', 'School', 'City', 'Zip', 'MailCity', 'MailZip', 'SOC', 'SOCType', 'EILCode', 'EILName', 'GSoffered', 'GSserved', 'Latitude']}, 'chain_of_thought_reasoning': 'The question asks for the highest eligible free rate for K-12 students in Alameda County. The hint provides the formula for calculating the eligible free rate: `Free Meal Count (K-12)` / `Enrollment (K-12)`. This information is found in the `frpm` table. The question also specifies a county, which is Alameda County. The `frpm` table contains the `County Name` and the `schools` table contains the `County` column. Therefore, we need both tables to filter by county and calculate the eligible free rate. The `frpm` table contains the necessary data for the calculation, and the `schools` table is needed to filter by county. The `satscores` table is not needed because the question does not involve SAT scores.', 'selected_tables': ['frpm', 'schools'], 'missing_table_status': 'success', 'missing_tables': [], 'missing_column_status': 'success', 'missing_columns': [], 'correct_columns': {'frpm': ['Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name']}, 'status': 'success'}, {'node_type': 'column_selection', 'tentative_schema': {'frpm': ['CDSCode', 'Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name'], 'schools': ['CDSCode', 'County']}, 'model_selected_columns': {'frpm': ['CDSCode', 'Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name'], 'schools': ['CDSCode', 'County']}, 'chain_of_thought_reasoning': "The question asks for the highest eligible free rate for K-12 students in Alameda County. The eligible free rate is calculated using 'Free Meal Count (K-12)' and 'Enrollment (K-12)' from the 'frpm' table. The question also specifies a county, which requires filtering by 'County Name' from the 'frpm' table. Therefore, we need to join the 'frpm' table with the 'schools' table on the common key 'CDSCode' to filter by county name.", 'missing_table_status': 'success', 'missing_tables': [], 'missing_column_status': 'success', 'missing_columns': [], 'correct_columns': {'frpm': ['Free Meal Count (K-12)', 'Enrollment (K-12)', 'County Name']}, 'status': 'success'}, {'node_type': 'candidate_generation', 'chain_of_thought_reasoning': "The question asks for the highest eligible free rate for K-12 students in Alameda County. The eligible free rate is calculated by dividing `Free Meal Count (K-12)` by `Enrollment (K-12)`. The table `frpm` contains the columns `Free Meal Count (K-12)` and `Enrollment (K-12)`. The table `schools` contains the `County` column. We need to join these two tables on `CDSCode` and filter by `County` = 'Alameda'. Then, calculate the eligible free rate, order by the rate in descending order, and limit to 1 to find the highest rate.", 'SQL': "SELECT T1.`Free Meal Count (K-12)` / T1.`Enrollment (K-12)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Alameda' AND T1.`Free Meal Count (K-12)` IS NOT NULL AND T1.`Enrollment (K-12)` IS NOT NULL ORDER BY T1.`Free Meal Count (K-12)` / T1.`Enrollment (K-12)` DESC LIMIT 1", 'status': 'success'}, {'node_type': 'evaluation', 'candidate_generation': {'exec_res': 1, 'exec_err': '--', 'Question': 'What is the highest eligible free rate for K-12 students in the schools in Alameda County?', 'Evidence': 'Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`', 'GOLD_SQL': "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1", 'PREDICTED_SQL': "SELECT T1.`Free Meal Count (K-12)` / T1.`Enrollment (K-12)` FROM frpm AS T1 INNER JOIN schools AS T2 ON T1.CDSCode = T2.CDSCode WHERE T2.County = 'Alameda' AND T1.`Free Meal Count (K-12)` IS NOT NULL AND T1.`Enrollment (K-12)` IS NOT NULL ORDER BY T1.`Free Meal Count (K-12)` / T1.`Enrollment (K-12)` DESC LIMIT 1"}, 'revision': {'exec_res': 'error', 'exec_err': "'NoneType' object is not subscriptable", 'Question': 'What is the highest eligible free rate for K-12 students in the schools in Alameda County?', 'Evidence': 'Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`', 'GOLD_SQL': "SELECT `Free Meal Count (K-12)` / `Enrollment (K-12)` FROM frpm WHERE `County Name` = 'Alameda' ORDER BY (CAST(`Free Meal Count (K-12)` AS REAL) / `Enrollment (K-12)`) DESC LIMIT 1", 'PREDICTED_SQL': '--'}, 'status': 'success'}]}}





CREATE TABLE frpm
(
        CDSCode TEXT not null primary key,
        `County Name` TEXT null, -- examples: `Alameda`
        `Enrollment (K-12)` REAL null, -- description: Enrollment (K-12) value description: K-12: 1st grade - 12nd grade
        `Free Meal Count (K-12)` REAL null, -- description: Free Meal Count (K-12) value description: eligible free rate = Free Meal Count / Enrollment
);

{'schools': {'AdmLName1': ['Free'], 'AdmLName2': ['Hughes'], 'City': ['Alameda'], 'MailCity': ['Alameda'], 'School': ['Preschool'], 'AdmFName1': ['Kate', 'Bree'], 'MailStreet': ['4600 Student Lane'], 'Street': ['4600 Student Lane'], 'GSoffered': ['K-12'], 'GSserved': ['K-12'], 'MailStrAbr': ['4600 Student Ln.'], 'StreetAbr': ['4600 Student Ln.'], 'EILName': ['Preschool'], 'County': ['Alameda'], 'AdmLName3': ['Yount'], 'District': ['Alameda Unified'], 'AdmFName3': ['Bree']}, 'frpm': {'School Name': ['Alameda High', 'Alameda County Community'], 'District Name': ['Alameda County Office of Education'], 'County Name': ['Alameda']}, 'satscores': {'sname': ['Alameda High'], 'cname': ['Alameda']}}


{'frpm': {'free meal count (k-12)': {'column_name': '', 'column_description': 'Free Meal Count (K-12)', 'value_description': 'eligible free rate = Free Meal Count / Enrollment'}, 'free meal count (ages 5-17)': {'column_name': '', 'column_description': 'Free Meal Count (Ages 5-17)', 'value_description': 'eligible free rate = Free Meal Count / Enrollment'}, 'frpm count (k-12)': {'column_name': '', 'column_description': 'Free or Reduced Price Meal Count (K-12)', 'value_description': 'eligible FRPM rate = FRPM / Enrollment'}, 'enrollment (k-12)': {'column_name': '', 'column_description': 'Enrollment (K-12)', 'value_description': 'K-12: 1st grade - 12nd grade'}}, 'satscores': {'enroll12': {'column_name': 'enrollment (1st-12nd grade)', 'column_description': 'enrollment (1st-12nd grade)', 'value_description': ''}}}



My engine_configs is like this, how to call the model indicated by model

def gemini_simple_api_call(model: str, prompt: str):
    client = genai.Client(
        vertexai=True,
        project=GCP_PROJECT,
        location=GCP_REGION,
    )
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
    )
    print(response.text, end="")
    return response.text


ENGINE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gemini-2.5-flash-lite": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-flash-lite",
            "temperature": 0,
            "top_p": 0.5,
            "top_k": 3,
            "seed": 1.0,
            "safety_settings": safety_settings
        }
    },
    "gemini-2.5-flash": {
        "constructor": VertexAI,
        "params": {
            "model": "gemini-2.5-flash",
            "temperature": 0,
            "top_p": 0.5,
            "top_k": 3,
            "seed": 1.0,
            "safety_settings": safety_settings
        }
    },
    ...


    ['Google has had its deal with Apple in place since 2003.\n\nWhen questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind. Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai, according to The Verge.\n\nGoogle understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their Google searches dropped 27%.\n\n“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.\n\nThe amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.\n\nGoogle agreed not to promote Chrome to Safari users\n\nAs part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with banners, pop-ups and other annoying means in other Google apps.\n\nThe agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.', 'We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.\n\nMeanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search — that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was making a big push to win business with people who wanted to buy identity theft protection.\n\nRevenue-shares to pre-install Google apps on Androids\n\nJamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.\n\nRosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.\n\nGoogle also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other companies to make or sell more Android devices, he said.\n\nExpedia complains of too many ads on search, expensive ad payments\n\nOn November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on organic listings.', 'The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.\n\nWhereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and it’s right there for everybody to read and to see plainly.\n\nI think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing, whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute. Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh antitrust scrutiny as Google.\n\nIt’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in the open, I should say — certainly they still wanted to keep them secret.\n\nBut I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they were similarly self-serving in terms of deals.\n\nI’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners, you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent. [Editor’s note: Netscape might feel differently.]', 'How ‘organic’ are Google’s results? This month, Wired posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates people’s search queries for maximum ad revenue. The example given was replacing a search for “children’s clothing” with “NIKOLAI-brand kidswear”. Loading Google rejected this in very strong terms, saying the piece was misleading and inaccurate while denying ever altering search terms. Wired removed the article for not meeting its standards, but the degree to which it was shared on social media and boosted in write-ups at other outlets shows just how willing people are to accept foul play on Google’s part. A suspicion of privacy invasion and inappropriate data handling follows the company at all times.\n\nGrey herself is a former vice-president at DuckDuckGo, a privacy-focused company founded explicitly to counter giants such as Google. She admits that she may have misinterpreted the evidence, but maintains that Google manipulates Search to maximise ad revenue. So what does Google say? In a post on X, Google’s official search liaison, Danny Sullivan, said ad systems do not affect the organic results, i.e. the list of blue links in search results that are not sponsored. “Ad keyword matching is a long-standing and well-known process that is designed to connect people to relevant ads. A separate process, which has nothing to do with ads, is used to match organic results to a query,” he said.', '“People don’t use Google because they have to, they use it because they want to,” he said. “This lawsuit simply ignores how intensely competitive and dynamic the technology industry is today.” Walker also points out that, when faced with a situation where Google is not the default, users tend to spend some effort putting Google back in charge. When Mozilla made Yahoo! the default on Firefox, most users changed it to Google. And when people set up a Windows device for the first time they frequently sidestep the Microsoft default — “Google” was the number one search query on Bing in 2021 — despite Microsoft making that very annoying to do. This is a compelling point, and many of us will be well accustomed to dodging all of Microsoft’s pleading while trying to get Chrome and Google set as defaults whenever we set up a new PC. But all of these arguments do ignore the fact that Google and its competitors are not on an even playing field. The question remains, why do users prefer Google? And if the answer, as Walker says, is that Google is simply better, the question becomes whether that’s down to its scale, its longevity, its ubiquity and all the user data it sucks up, which no other company could possibly match.\n\nFor the US, which has to prove that Google somehow broke the law to achieve the status quo, this legal case could be an uphill battle. But for the rest of us, it offers a rare opportunity for some insight into what the tech giant does with its enormous market share. Is it still focused on being the best? Or, as has been alleged, does it use its ubiquity to squeeze us for money even at the expense of product quality? How ‘organic’ are Google’s results? This month, Wired posted an opinion article by lawyer and privacy advocate Megan Gray, which alleged that Google had accidentally revealed during the trial that it manipulates people’s search queries for maximum ad revenue.', 'How did Epic’s argument fare against Apple?\n\nWell… both sides lost! But Epic arguably lost more. Even though Apple has incredible power over the iPhone, Judge Yvonne Gonzalez Rogers ruled the company didn’t have an unfair monopoly in this case – partly because she decided the relevant market for Fortnite was “digital mobile gaming transactions” rather than, say, iPhone apps. She also decided that Epic violated its developer agreement with Apple and would have to pay.\n\nBut she also barred Apple from keeping iPhone users in the dark about alternate ways they can pay for apps – and may have even allowed developers to add their own payment mechanisms. I won’t dwell on the Apple ruling, as I’m ethically bound, but my colleague Adi Robertson has a comprehensive breakdown.\n\nHow can Epic possibly have a case against Google when Apple already won?\n\nEpic declined to answer this on the record, among other questions we asked... but three things to consider:\n\n1) That was a different case. Also, that one’s not over till the Supreme Court weighs in or declines to take a look.\n\n2) Google can’t tell jurors that Apple won its case, or that other plaintiffs settled. The judge in this case says so!\n\n3) Oh, and by the way, this is a jury trial.\n\nWait, why does that matter?\n\nEpic and Google have to convince a jury, not the judge, which is totally different from how the Apple case played out. (That one was a “bench trial.”)\n\nMaybe all the evidence of tricky deals inside Google might sway a jury against the company? Maybe Google scaremongering that sideloaded apps equal gaping security holes will sway a jury against Epic instead? Who knows!\n\n(If you’re a Epic juror reading this — stop! Judge Donato explicitly said you’re in a “news-free bubble” through mid-December, folks.)\n\nDidn’t other parties sue Google too?\n\nThey did! And then, they all settled.', 'law. The Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.\n\nThis isn’t Google’s only antitrust case in trial right now. The search engine giant last week settled a separate antitrust lawsuit with dating site Match Group. On November 6, Google went to trial with Fortnite maker Epic Games. The latter hopes to prove that Google engages in anticompetitive behavior with regard to its Android app store, Google Play, and its commission structure.\n\nNow, onto the roundup!\n\nA window into Google’s most popular search queries\n\nJudge Amit Mehta ruled to make a list public that provides a glimpse of which search terms make Google the most money. The list of popular search terms ordered by revenue includes 20 terms that were lucrative for the week of September 22, 2018. Information like revenue per search term, how many queries each of those terms got, along with a separate list of popular search terms ordered by queries (not revenue), were all redacted. The list we can see is as follows:\n\niPhone 8\n\niPhone 8 plus\n\nAuto insurance\n\nCar insurance\n\nCheap flights\n\nCar insurance quotes\n\nDirecTV\n\nOnline colleges\n\nAT&T\n\nHulu\n\niPhone\n\nUber\n\nSpectrum\n\nComcast\n\nXfinity\n\nInsurance quotes\n\nFree credit report\n\nCheap car insurance\n\nAARP\n\nLifeLock\n\nThere is, in reality, little surprise here.', 'More from the US v Google trial: Vertical search, pre-installs and the case of Firefox/Yahoo\n\nWe’re nearly two months into the Justice Department’s landmark antitrust case against Google — one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the 1990s — and the revelations just keep getting juicier.\n\nIn our last roundup, we learned how Google spent $26.3 billion in 2021 making itself the default search engine across platforms and how Google tried to have Chrome preinstalled on iPhones. Over the past couple of weeks, more of the inner workings of Google has come to light, including some of the search engine’s most lucrative search queries, what the revenue-share agreements between Google and Android OEMs look like and why Expedia has a bone to pick with Google.\n\nBefore we go into some of these tidbits…\n\nWhy the Google vs. U.S. antitrust case matters\n\nThe government has argued that Google uses its platforms and deals with partners to block out any competition in search or advertising, thus hindering competitors from accessing the data they’d need to improve their products. If Judge Amit Mehta rules against Google, the search giant may have to change its behavior and share its APIs with third-party developers. It may also be banned from making anticompetitive and exclusive deals with smartphone and computer manufacturers and wireless carriers. Google might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday. Enforcers want to show that antitrust law remains relevant and that even though Google is basically the God of the internet, it’s still no match for the U.S. law. The Google outcome could also have a ripple effect on other Big Tech cases.', 'Back then, Google’s legal chief David Drummond sent Microsoft an angry letter, saying that making Internet Explorer the search default was anticompetitive. Oh, how the tables have turned.\n\nAfter establishing that Google understands the inherent value of defaults, Bellshaw brought up Drummond’s letter to establish the hypocrisy of Google today. The letter declared that problems with a default setting are made worse by how changes to defaults are handled, and that most end users “do not change defaults.”\n\nThese are exactly the arguments that other search engine companies, like DuckDuckGo, Brave or Microsoft’s Bing, make when they accuse Google of being anticompetitive by making deals with Apple and others. The DOJ doubled down on this, saying Google has become the monopoly it denounced years ago.\n\nWhat does it all mean?\n\nThe case is expected to continue for several weeks, bringing to a head one of the biggest fights in tech antitrust since the U.S. took Microsoft to trial in the 1990s.\n\nIf the judge rules against Google, the outcome could look a lot like the Microsoft deal, in which the computer company was required to change its behavior and share its APIs with third-party developers. Microsoft was also banned from making anticompetitive and exclusive deals with computer manufacturers.\n\nGoogle might end up having to turn over all or most of the data it has collected to other search engines so they can improve their products and attract more users. The DOJ has said that Google gets 16 times more data than Bing does everyday.\n\nThe Google outcome could also have a ripple effect on other Big Tech cases. The FTC sued Amazon in September for using anticompetitive and unfair strategies to illegally maintain its monopoly power. The DOJ has been investigating Apple for years over the company’s policy for third-party apps on its devices and whether it unfairly favors its own products. There’s an ongoing case between the FTC and Facebook, wherein the agency calls on Facebook to sell Instagram and WhatsApp.']



    ['We’ve already established that Google and Apple have a long and mutually beneficial relationship, even while competing, so it’s not surprising to see three Apple search-related queries bringing in the big bucks — not least since September 22, 2017 was the official release date of the iPhone 8.\n\nMeanwhile, queries like “car insurance,” “cheap flights” and “credit report” are perennial favorites and they speak to how much Google dominates vertical search — that is, search in very specific market categories. As for LifeLock… the big Equifax data breach of 2017 was a hot topic in September 2017 and LifeLock was making a big push to win business with people who wanted to buy identity theft protection.\n\nRevenue-shares to pre-install Google apps on Androids\n\nJamie Rosenberg, a Google employee who focuses on Android and Google Play, testified in Google’s defense on November 8. He said that the competition between Google and Apple is “as intense as it gets,” reports Bloomberg.\n\nRosenberg explained how Google gets manufacturers to sign a mobile app distribution agreement (MADA) that requires Android smartphone makers (like Samsung or Oppo) to pre-load a bundle of 11 Google apps on the device, including Search, Chrome and Play. They don’t have to be the default choices, he said.\n\nGoogle also has revenue share agreements (RSAs) with smartphone makers and wireless carriers (like Verizon) that require them to set Google search and Chrome web browser as defaults. Rosenberg defended the move and said it was because Google apps [like Search] are “best in class.” The RSAs also motivate other companies to make or sell more Android devices, he said.\n\nExpedia complains of too many ads on search, expensive ad payments\n\nOn November 1, Barry Diller, chair of Expedia and IAC, testified about his concerns regarding the increasing number of ads in search results having an impact on organic listings.', 'Google has had its deal with Apple in place since 2003.\n\nWhen questioned on the amount of money Google spends to get first pick of search engines, Pichai said that the decision was made with the consumer in mind. Google pays big bucks to be everywhere so that it can take in all the data and be the best search engine across different companies’ devices, said Pichai, according to The Verge.\n\nGoogle understood the value of defaults very early on. U.S. Justice Department lawyer Meagan Bellshaw showed Pichai a 2007 email from a Google product strategy meeting containing data showing that when people changed their browser homepage to Google, they did 15% more Google searches. When they switched away, their Google searches dropped 27%.\n\n“Nitin argues that focusing on homepage market share is one of the most effective things we can do to make gains in search market share,” read an email that summarized the meeting and was sent to Pichai, as well as other Google leaders, according to The Verge.\n\nThe amount that Google spent on homepage market share has been a fixing point in the trial. Earlier this month, the CEOs of Microsoft and DuckDuckGo testified that their search engines would have been far more successful, even competitive with Google, had they been able to make similar deals with Apple. Microsoft CEO Satya Nadella even said he was willing to spend $15 billion per year to get Bing into Apple’s default search, per The Information.\n\nGoogle agreed not to promote Chrome to Safari users\n\nAs part of its search deal with Apple, Google agreed not to promote Chrome to Safari users, reports Bloomberg. Google would have been able to do this with banners, pop-ups and other annoying means in other Google apps.\n\nThe agreement also meant that Apple never switched to a Google competitor or allowed users to choose their browser when setting up their iPhones.', 'But the questions put to Cue were the same ones the DOJ is going to keep asking: is Google really the best search engine, or is it just the one writing the biggest checks? And if those checks went away, what would the search engine market look like? Cue said Apple’s never really thought about it. Google said Apple would be silly to do so. And the Justice Department thinks it’s about time Apple starts doing so.', '“Google pays actual and potential competitors not to compete. Literally gives them money and other things of value,” said Bornstein. “It’s like Google saying, ‘Here’s $360 million’ — that’s an actual number you’ll hear about — why don’t you sit this one out and let me win?”\n\nThe upshot for consumers, Epic’s earlier legal filings have suggested, is that we pay higher prices for apps than we would if there were more competition and / or lower app store and payment processing fees. But while this will probably come up later in the trial, Epic chose to focus more on simply painting Google as the bad guy on day one.\n\nIt’s not clear how much of that evidence will hold up on closer examination. That $360 million, for instance, refers to an alleged payment that kept Activision from opening an app store that could compete with Google Play. But Activision told The Verge in 2022 that it “never entered into an agreement that Activision would not open its own app store” — and Google is now, it says, armed with the evidence to prove it. On Monday, Epic’s attorney admitted Google “was too clever” to draw up contracts that specifically forced developers not to compete with the Play Store. The overall narrative is compelling, though — and I’m not sure Google’s opening statement countered it. Google spent its 45 minutes attempting to explain that its dominance over the Android app market isn’t anything nefarious but simply the natural outcome of Google fiercely competing with the iPhone and its iOS App Store, where Google would like the court to believe that competition truly lies.\n\nIf Google can convince the jury of that, it could be a winning argument in the case — because obviously, Google doesn’t have a monopoly on app stores or phones in general. “You cannot separate the quality of a phone from the quality of the apps in its app store, and that means Google and Apple compete against each other,” began Google lead attorney Glenn Pomerantz.', 'However, Google will counter Epic’s arguments by pointing out that it has rolled out a new option for app developers called User Choice Billing, which Epic has declined to use. This program, still in pilot testing, is open to all developers who sell apps in the 35 markets where it’s now available, including the U.S., and reduces the standard commission by 4% for companies who choose to use their own payment processing solution. Spotify and Bumble were the initial testers for the new system, first introduced in November 2022, and Match, as part of its settlement agreement, will also now take advantage of this option.\n\nGoogle will also argue that its commissions aren’t just tied to payment processing, but rather help it to develop other features and controls to keep Android users safe and guide them to discovering new applications, as well as help fund the tools and services developers need to build and grow their apps. Google also helps developers reach a broader audience beyond the mobile phone, with support for Android across platforms, ranging from tablets to TVs to autos and more. And it will note that 99% of its developers qualify for a commission of 15% or less, based on their revenues.\n\nThe competitive landscape with other app stores, OEMs\n\nThe tech giant additionally plans to push back at Epic’s claims that it doesn’t have competition in the app store business. It will point out that not only does the Play Store compete with Apple’s App Store, which the Ninth Circuit ruling in the Apple case agreed upon, too, but Android also supports competitive app stores like Samsung’s Galaxy Store and Amazon’s Appstore.\n\nEpic, however, will attempt to demonstrate that Google makes it hard for alternative app stores to succeed and reach consumers, noting that 90% of apps are still downloaded through Google Play. And it will point out that Google bundles the Play Store with other Google apps that Android OEMs (original equipment manufacturers) have to agree to in order to use Android.', 'The thing with Apple is all of their antitrust trickery is internal to the company. They use their store, their payments, they force developers to all have the same terms, they force OEMs and carriers to all have the same terms.\n\nWhereas Google, to achieve things with Android, they were going around and paying off game developers, dozens of game developers, to not compete. And they’re paying off dozens of carriers and OEMs to not compete — and when all of these different companies do deals together, lots of people put things in writing, and it’s right there for everybody to read and to see plainly.\n\nI think the Apple case would be no less interesting if we could see all of their internal thoughts and deliberations, but Apple was not putting it in writing, whereas Google was. You know, I think Apple is... it’s a little bit unfortunate that in a lot of ways Apple’s restrictions on competition are absolute. Thou shalt not have a competing store on iOS and thou shalt not use a competing payment method. And I think Apple should be receiving at least as harsh antitrust scrutiny as Google.\n\nIt’s interesting to me that because Google distributes the Android operating system as open source, they had to put all these deals out in the open. More out in the open, I should say — certainly they still wanted to keep them secret.\n\nBut I’m going down my story about all the best emails from the Epic v. Apple trial — and we do have a lot of documents from both Apple and Google that show they were similarly self-serving in terms of deals.\n\nI’d say this is the thing that’s disappointed me the most with Apple and Google: even at the peak of the antitrust trial against Microsoft, Microsoft was awesome to developers. Microsoft has always been awesome to developers, always being respectful, giving developers a great deal and treating them as partners, you know? And so even as Microsoft was crushing corporate competitors, the developer experience was excellent. [Editor’s note: Netscape might feel differently.]']