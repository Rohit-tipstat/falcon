import os
import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from openai import OpenAI
from langsmith import Client as LangSmithClient
from langsmith import traceable
from langsmith.wrappers import wrap_openai
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('waste_composition_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Pydantic models
class WasteComposition(BaseModel):
    composition_name: str
    composition_percentage: float

class WasteCompositionResponse(BaseModel):
    composition_dict: list[WasteComposition]
    citation_dict: list[str]

# Initialize OpenAI client
try:
    openai_key = os.environ['OPENAI_API_KEY']
    client = wrap_openai(OpenAI(api_key=openai_key))
    logger.info("OpenAI client initialized successfully")
except KeyError:
    logger.error("OPENAI_API_KEY not found in environment variables")
    raise Exception("OPENAI_API_KEY not found in environment variables. Ensure it is set in .env file.")

# Validate and initialize LangSmith client
try:
    langchain_api_key = os.environ['LANGCHAIN_API_KEY']
    if not langchain_api_key:
        raise KeyError("LANGCHAIN_API_KEY is empty")
    langsmith_client = LangSmithClient()
    logger.info("LangSmith client initialized successfully")
except KeyError:
    logger.error("LANGCHAIN_API_KEY not found or empty in environment variables")
    raise Exception("LANGCHAIN_API_KEY not found or empty. Ensure it is set in .env file.")
except Exception as e:
    logger.error(f"Failed to initialize LangSmith client: {str(e)}")
    raise Exception(f"Failed to initialize LangSmith client: {str(e)}")

# FastAPI app
app = FastAPI(
    title="Waste Composition API",
    description="API for retrieving municipal solid waste composition data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@traceable(run_type="chain", project_name="waste_composition_api")
async def get_waste_composition(area: str) -> WasteCompositionResponse:
    """Core function to fetch waste composition data"""
    logger.info(f"Processing waste composition request for area: {area}")
    try:
        # Make API call to OpenAI
        response = client.responses.create(
            model="gpt-4o",
            tools=[{
                "type": "web_search_preview",
                "search_context_size": "high",
            }],
            input=f"""What is the composition of Muncipal Solid Waste for area {area}? I need the composition in percentage. Make sure you always provide the correct sources link from where you have extracted the informations.
            Make sure you avoid extracting information from government websites, rely on latest research papers, university research, and other reliable sources.
            The composition always should include the following elements only: paper and paper board materials, glass, metals, plastics, yard trimmings, food, wood, rubber, leather, textiles and other materials.
            The total composition should be 100% and the sum of all the elements should be equal to 100%. If the information is not available for the given area, please provide the information for the nearest area.
            If the zipcode is not a valid zipcode then return empty list.
            If the composition then return a message as "No information available for the given zipcode" 
            """,
        )
        structured_result = response.output[1].content[0].text
        logger.debug(f"Raw OpenAI response: {structured_result}")

        # Format the response
        response_formatted = client.responses.parse(
            model="gpt-4o-mini",
            input="""You are responsible for extracting the following elements and it composition.\n
             Elements to extract with its percentage:\n
               1. paper and paper board materials\n 
               2. glass \n
               3. metals \n
               4. plastics \n
               5. yard trimmings \n
               6. food \n
               7. wood \n
               8. rubber \n
               9. leather \n
               10. textiles \n
               11. Construction & demolition debris	\n
               12. electronic waste \n
               13. others(hazardous, diapers, etc.) \n

            The composition element can be sometimes named some thing else. make sure you extrcat it if falls under the above category.\n
            The total composition should be 100% and the sum of all the elements should be equal to 100%.\n
             The text you need to extract from is given below:\n""" + structured_result,
            text_format=WasteCompositionResponse,
        )
        
        dictionary_link = []
        if len(response.output[1].content[0].annotations) > 0:
            for i in range(len(response.output[1].content[0].annotations)):
                dictionary_link.append(response.output[1].content[0].annotations[i].url)
            logger.info(f"Extracted {len(dictionary_link)} citations for area {area}")

        dictionary_composition = dict()
        if len(response_formatted.output_parsed.composition_dict) > 0:
            for i in range(len(response_formatted.output_parsed.composition_dict)):
                dictionary_composition[response_formatted.output_parsed.composition_dict[i].composition_name] = response_formatted.output_parsed.composition_dict[i].composition_percentage
            logger.info(f"Extracted composition data for {len(dictionary_composition)} materials for area {area}")

        # Validate total percentage
        total_percentage = sum(dictionary_composition.values())
        response_data = {
            "output": structured_result,
            "citations": dictionary_link,
            "composition": dictionary_composition
        }
        if abs(total_percentage - 100.0) > 0.03:
            logger.warning(f"Total percentage {total_percentage}% does not sum to 100% for area {area}")
            return [response_data]

        logger.info(f"Successfully processed composition data for area {area}")
        return [response_data]

    except Exception as e:
        logger.error(f"Error processing area {area}: {str(e)}")
        return WasteCompositionResponse(
            citation_dict=[],
            message=f"Error processing request: {str(e)}"
        )

@app.get("/waste-composition/{area}")
async def waste_composition_endpoint(area: str):
    """Endpoint to get waste composition for a given area"""
    logger.info(f"Received request for area: {area}")
    try:
        result = await get_waste_composition(area)
        logger.info(f"Successfully processed request for area {area}")
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content=result
        )
        
    except Exception as e:
        logger.error(f"Internal server error for area {area}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return {"status": "healthy", "version": "1.0.0"}
