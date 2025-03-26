# Agentic Python Code Structure and Dependencies Analyzer using CrewAI Tools (without langchain)

# Install dependencies
# pip install crewai crewai-tools python-dotenv

import json
import os
import glob
from crewai import LLM, Agent, Task, Crew, Process
from crewai_tools import FileReadTool
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# Initialize LLM using CrewAI's built-in OpenAI tool (no direct langchain dependency)
llm = LLM(model="openai/gpt-4o-mini")

# Specify workspace directory
workspace_folder = r"C:\Users\aubry\Documents\Programs\Taxo_Analysis_Interface_App\taxo_analysis_interface_app"  # Update this path to your target directory

# Collect Python files from workspace
python_files = glob.glob(os.path.join(workspace_folder, "**/*.py"), recursive=True)

# Initialize file read tool
file_read_tool = FileReadTool()

# Define Agents
code_reader_agent = Agent(
    role="Python Code Parser",
    goal="""Systematically read and parse Python files to extract detailed code structure information:
    1. Extract module-level docstrings and comments
    2. Identify all class and function definitions with their signatures
    3. Capture imports and module dependencies
    4. Preserve code hierarchy and nesting relationships
    5. Document code organization and file structure""",
    backstory="""You are an expert Python code parser specialized in breaking down source code into well-structured components. 
    You carefully analyze file contents to create organized representations that preserve both syntax and semantic relationships.
    You pay special attention to docstrings, type hints, and code organization patterns.""",
    tools=[file_read_tool],
    llm=llm,
    verbose=True
)

structure_analyzer_agent = Agent(
    role="Structure Analyzer",
    goal="""Analyze code structure and relationships to:
    1. Map class hierarchies, inheritance patterns, and composition relationships
    2. Document class attributes, methods, and their visibility (public/private)
    3. Identify key design patterns and architectural components
    4. Trace object interactions and dependencies between classes
    5. Evaluate code modularity and coupling between components""",
    backstory="""You are an expert software architect specializing in Python code analysis. 
    You excel at understanding complex object-oriented structures, identifying relationships between components,
    and mapping how different parts of the application interact. You focus on both the technical implementation 
    details and the higher-level architectural patterns.""",
    llm=llm,
)

# dependency_analyzer_agent = Agent(
#     role="Dependency Analyzer",
#     goal="Identify internal and external dependencies, including modules, libraries, and inter-file dependencies.",
#     backstory="You excel at tracing imports, usage patterns, and external library calls in Python code.",
#     llm=llm,
# )

# dependency_analyzer_agent = Agent(
#     role="Dependency Analyzer",
#     goal="""Analyze and document dependencies with focus on:
#     1. Map all internal module dependencies and import relationships
#     2. Identify external library dependencies and their usage patterns
#     3. Create PlantUML component diagrams showing module interactions
#     4. Document dependency cycles and suggest improvements
#     5. Analyze the coupling between different components""",
#     backstory="""You are an expert dependency analyst specializing in Python applications.
#     You excel at mapping complex dependency relationships and can express them clearly using PlantUML diagrams.
#     Your analysis helps teams understand and optimize their application's structure.""",
#     llm=llm,
# )

report_writer_agent = Agent(
    role="Technical Documentation Specialist",
    goal="""Create comprehensive technical documentation including:
    1. Generate detailed PlantUML class diagrams showing class relationships
    2. Create PlantUML sequence diagrams for key interactions
    3. Provide a clear hierarchical structure of the codebase
    4. Document architectural patterns and design decisions
    5. Include dependency graphs and component relationships""",
    backstory="""You are an expert technical writer with deep knowledge of UML and documentation best practices.
    You excel at creating clear, visual documentation using PlantUML syntax and markdown.
    You know how to organize complex technical information into easily digestible formats.""",
    llm=llm,
)

# Define Tasks

# analyze_dependencies_task = Task(
#     description="Analyze dependencies within the code, listing internal modules, external libraries, and imports.",
#     agent=dependency_analyzer_agent,
#     expected_output="List of dependencies, both internal and external, with details on imports.",
#     context=[read_code_task],
# )
read_code_task = Task(
    description=f"""Systematically parse and extract information from these Python files: {python_files}
    Focus on:
    1. Module-level docstrings and comments
    2. Class and function definitions with signatures
    3. Import statements and dependencies
    4. Code hierarchy and nesting
    5. File organization patterns""",
    agent=code_reader_agent,
    expected_output="Structured representation of code components with preserved relationships and documentation.",
)

analyze_structure_task = Task(
    description="""Analyze the parsed code to create a comprehensive structural map:
    1. Document class hierarchies and inheritance patterns
    2. Map class attributes, methods, and their visibility
    3. Identify implemented design patterns
    4. Trace object interactions and dependencies
    5. Evaluate component coupling and modularity""",
    agent=structure_analyzer_agent,
    expected_output="Detailed architectural analysis with class relationships, patterns, and component interactions.",
    context=[read_code_task],
)

write_report_task = Task(
    description="""Generate comprehensive technical documentation using PlantUML:
    1. Create class diagrams showing relationships and hierarchies
    2. Design sequence diagrams for key interactions
    3. Document the codebase hierarchy
    4. Detail architectural patterns
    5. Include dependency and component relationships""",
    agent=report_writer_agent,
    expected_output="Complete technical documentation with UML diagrams, architectural overview, and relationship maps.",
    context=[analyze_structure_task],
)

# Assemble Crew
code_analysis_crew = Crew(
    agents=[
        code_reader_agent,
        structure_analyzer_agent,
        # dependency_analyzer_agent,
        report_writer_agent,
    ],
    tasks=[read_code_task, analyze_structure_task, write_report_task],
    process=Process.sequential,
)

# Run the Crew
result = code_analysis_crew.kickoff()

# print(result)
# Write the results to a markdown file
with open('code_analysis_report.md', 'w', encoding='utf-8') as f:
    f.write('# Code Analysis Report\n\n')
    f.write(result.raw)
# Save JSON output if available
if result.json_dict:
    with open('code_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(result.json_dict, f, indent=2)
    print("JSON output saved to code_analysis_report.json")

# Save Pydantic output if available
if result.pydantic:
    with open('code_analysis_report_pydantic.txt', 'w', encoding='utf-8') as f:
        f.write(str(result.pydantic))
    print("Pydantic output saved to code_analysis_report_pydantic.txt")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'task_output_{timestamp}.txt', 'w', encoding='utf-8') as f:
    f.write(f"Tasks Output:\n{result.tasks_output}\n")
    f.write(f"\nToken Usage:\n{result.token_usage}")