"""
FastAPI REST API for ShelfScale weight extraction and nutrition scoring
"""

import os
import io
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Union, Any
from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
import logging

# Import ShelfScale components
from shelfscale.data_processing.weight_extraction import WeightExtractor, predict_missing_weights, load_density_map
from shelfscale.scoring import score_traffic_lights, score_nutri
from shelfscale.main import build_nutrient_view, apply_nutrition_scoring

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="ShelfScale API",
    description="REST API for food product weight extraction and nutrition scoring",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize global components
try:
    weight_extractor = WeightExtractor(target_unit='g')
    density_map = load_density_map()
    logger.info("Initialized weight extractor and density map")
except Exception as e:
    logger.error(f"Error initializing components: {e}")
    weight_extractor = None
    density_map = None

# Request/Response Models

class WeightExtractItem(BaseModel):
    """Item for weight extraction"""
    text: str = Field(..., description="Text containing weight information")
    category: Optional[str] = Field(None, description="Food category for density lookup")
    super_category: Optional[str] = Field(None, description="Super category for density lookup")

class WeightExtractRequest(BaseModel):
    """Request for weight extraction"""
    items: List[Union[str, WeightExtractItem]] = Field(..., description="List of texts or objects to extract weights from")

class WeightResult(BaseModel):
    """Weight extraction result"""
    weight_g: Optional[float] = Field(None, description="Extracted weight in grams")
    unit: Optional[str] = Field(None, description="Original unit")
    source: str = Field(..., description="Source of weight extraction")
    confidence: float = Field(..., description="Extraction confidence (0-1)")
    raw_text: str = Field(..., description="Original input text")

class WeightExtractResponse(BaseModel):
    """Response for weight extraction"""
    results: List[WeightResult] = Field(..., description="Weight extraction results")

class ProductScoreItem(BaseModel):
    """Product for nutrition scoring"""
    name: str = Field(..., description="Product name")
    weight_text: Optional[str] = Field(None, description="Weight text for extraction")
    normalized_weight_g: Optional[float] = Field(None, description="Pre-normalized weight in grams")
    category: Optional[str] = Field(None, description="Food category")
    super_category: Optional[str] = Field(None, description="Super category")
    
    # Nutrition data (per 100g/ml)
    energy_kcal: Optional[float] = Field(None, description="Energy in kcal per 100g")
    energy_kj: Optional[float] = Field(None, description="Energy in kJ per 100g")
    fat_g: Optional[float] = Field(None, description="Fat in grams per 100g")
    saturated_fat_g: Optional[float] = Field(None, description="Saturated fat in grams per 100g")
    sugars_g: Optional[float] = Field(None, description="Sugars in grams per 100g")
    salt_g: Optional[float] = Field(None, description="Salt in grams per 100g")
    sodium_mg: Optional[float] = Field(None, description="Sodium in mg per 100g")
    fiber_g: Optional[float] = Field(None, description="Fiber in grams per 100g")
    protein_g: Optional[float] = Field(None, description="Protein in grams per 100g")

class ScoreRequest(BaseModel):
    """Request for nutrition scoring"""
    items: List[ProductScoreItem] = Field(..., description="List of products to score")

class ProductScore(BaseModel):
    """Product nutrition score result"""
    name: str = Field(..., description="Product name")
    normalized_weight_g: Optional[float] = Field(None, description="Normalized weight in grams")
    
    # Traffic Lights
    traffic_lights: Optional[Dict[str, str]] = Field(None, description="Traffic light scores")
    
    # Nutri-Score
    nutri_score: Optional[int] = Field(None, description="Nutri-Score value")
    nutri_grade: Optional[str] = Field(None, description="Nutri-Score grade (A-E)")
    
    # Metadata
    confidence: float = Field(..., description="Overall scoring confidence (0-1)")
    provenance: Dict[str, str] = Field(..., description="Data source provenance")

class ScoreResponse(BaseModel):
    """Response for nutrition scoring"""
    results: List[ProductScore] = Field(..., description="Nutrition scoring results")

# API Endpoints

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "ShelfScale API",
        "version": "1.0.0",
        "endpoints": {
            "weight_extraction": "/weights/extract",
            "nutrition_scoring": "/products/score",
            "batch_scoring": "/batch/score",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = "healthy" if weight_extractor is not None else "degraded"
    return {
        "status": status,
        "components": {
            "weight_extractor": weight_extractor is not None,
            "density_map": density_map is not None
        }
    }

@app.post("/weights/extract", response_model=WeightExtractResponse)
async def extract_weights(request: WeightExtractRequest):
    """
    Extract weights from text descriptions
    
    Supports both simple text strings and structured objects with category information
    for density-based volume→weight conversion.
    """
    if weight_extractor is None:
        raise HTTPException(status_code=503, detail="Weight extractor not available")
    
    results = []
    
    for item in request.items:
        try:
            # Handle both string and object inputs
            if isinstance(item, str):
                text = item
                row = None
            else:
                text = item.text
                # Create a pandas Series for density lookup
                row_data = {}
                if item.category:
                    row_data['Food_Category'] = item.category
                if item.super_category:
                    row_data['Super_Category'] = item.super_category
                row = pd.Series(row_data) if row_data else None
            
            # Extract weight
            weight, unit = weight_extractor.extract_from_text(text, row)
            
            # Calculate confidence based on extraction success and specificity
            confidence = 0.0
            if weight is not None:
                confidence = 0.8  # Base confidence for successful extraction
                if unit and unit in weight_extractor.weight_units:
                    confidence = 0.9  # Higher confidence for weight units
                elif unit and unit in weight_extractor.volume_units and row is not None:
                    confidence = 0.7  # Lower confidence for volume→weight conversion
            
            # Determine source
            source = "extracted"
            if weight is not None and unit in weight_extractor.volume_units:
                source = "volume_conversion"
            
            results.append(WeightResult(
                weight_g=weight,
                unit=unit,
                source=source,
                confidence=confidence,
                raw_text=text
            ))
            
        except Exception as e:
            logger.error(f"Error extracting weight from '{text}': {e}")
            results.append(WeightResult(
                weight_g=None,
                unit=None,
                source="error",
                confidence=0.0,
                raw_text=text if isinstance(item, str) else item.text
            ))
    
    return WeightExtractResponse(results=results)

@app.post("/products/score", response_model=ScoreResponse)
async def score_products(request: ScoreRequest):
    """
    Calculate nutrition scores for products
    
    Supports both Traffic Light and Nutri-Score calculations based on
    provided nutrition data. Missing nutrition data will be estimated
    where possible.
    """
    results = []
    
    # Convert request items to DataFrame
    df_data = []
    for item in request.items:
        row_data = {
            'Food_Name': item.name,
            'Food_Category': item.category,
            'Super_Category': item.super_category,
            'Energy_kcal': item.energy_kcal,
            'Energy_kJ': item.energy_kj,
            'Fat_g': item.fat_g,
            'SatFat_g': item.saturated_fat_g,
            'Sugars_g': item.sugars_g,
            'Salt_g': item.salt_g,
            'Sodium_mg': item.sodium_mg,
            'Fiber_g': item.fiber_g,
            'Protein_g': item.protein_g,
            'Normalized_Weight': item.normalized_weight_g
        }
        
        # Extract weight from text if provided
        if item.weight_text and weight_extractor:
            try:
                # Create row for density lookup
                lookup_row = pd.Series({
                    'Food_Category': item.category or '',
                    'Super_Category': item.super_category or ''
                })
                weight, unit = weight_extractor.extract_from_text(item.weight_text, lookup_row)
                if weight is not None and row_data['Normalized_Weight'] is None:
                    row_data['Normalized_Weight'] = weight
            except Exception as e:
                logger.warning(f"Error extracting weight for {item.name}: {e}")
        
        df_data.append(row_data)
    
    df = pd.DataFrame(df_data)
    
    # Apply scoring
    try:
        scored_df = apply_nutrition_scoring(df, 'all')
        
        # Convert results back to response format
        for idx, row in scored_df.iterrows():
            # Build traffic lights dict
            traffic_lights = None
            if 'Traffic_Lights_Summary' in scored_df.columns:
                traffic_lights = {
                    'fat': row.get('Traffic_Lights_Fat', 'unknown'),
                    'saturated_fat': row.get('Traffic_Lights_SatFat', 'unknown'),
                    'sugars': row.get('Traffic_Lights_Sugars', 'unknown'),
                    'salt': row.get('Traffic_Lights_Salt', 'unknown'),
                    'summary': row.get('Traffic_Lights_Summary', 'unknown')
                }
            
            # Calculate confidence
            confidence = row.get('Score_Confidence', 0.5)
            
            # Build provenance info
            provenance = {
                'nutrition_source': row.get('Nutrient_Source', 'unknown'),
                'weight_source': row.get('Weight_Source', 'unknown')
            }
            
            results.append(ProductScore(
                name=row['Food_Name'],
                normalized_weight_g=row.get('Normalized_Weight'),
                traffic_lights=traffic_lights,
                nutri_score=row.get('Nutri_Score'),
                nutri_grade=row.get('Nutri_Grade'),
                confidence=confidence,
                provenance=provenance
            ))
            
    except Exception as e:
        logger.error(f"Error scoring products: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")
    
    return ScoreResponse(results=results)

@app.post("/batch/score")
async def batch_score_file(file: UploadFile = File(...)):
    """
    Score products from uploaded CSV file
    
    Accepts a CSV file with product data and returns a CSV file with
    nutrition scores added.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a CSV")
    
    try:
        # Read uploaded CSV
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Apply scoring
        scored_df = apply_nutrition_scoring(df, 'all')
        
        # Convert back to CSV
        output = io.StringIO()
        scored_df.to_csv(output, index=False)
        output.seek(0)
        
        # Return as streaming response
        response = StreamingResponse(
            io.BytesIO(output.getvalue().encode('utf-8')),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=scored_{file.filename}"}
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing batch file: {e}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Utility function for running the API
def run_api(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    Run the FastAPI server
    
    Args:
        host: Host to bind to
        port: Port to bind to  
        debug: Enable debug mode
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info" if not debug else "debug")

if __name__ == "__main__":
    # Run the API server
    run_api(debug=True)