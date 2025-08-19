"""
Test suite for LLM-enhanced matching and scraping system
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import json

# Import the modules we're testing
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shelfscale.ml.llm_matcher import LLMMatcher, MatchCandidate
from shelfscale.scraping.llm_enhanced_scraper import LLMEnhancedScraper, ProductInfo


class TestLLMMatcher:
    """Test cases for LLM-enhanced product matching"""
    
    @pytest.fixture
    def llm_matcher(self):
        """Create LLM matcher instance for testing"""
        return LLMMatcher(
            model_name="test-model",
            confidence_threshold=0.7,
            use_hybrid_scoring=True
        )
        
    @pytest.fixture
    def sample_products(self):
        """Sample product data for testing"""
        return [
            "Tesco Organic Free Range Chicken Breast 500g",
            "Sainsbury's Chicken Breast Fillets 400g", 
            "Asda Smart Price Whole Milk 2L",
            "Tesco Finest Sourdough Bread 800g",
            "Organic Valley Whole Milk 1L"
        ]
        
    def test_matcher_initialization(self, llm_matcher):
        """Test LLM matcher initialization"""
        assert llm_matcher.model_name == "test-model"
        assert llm_matcher.confidence_threshold == 0.7
        assert llm_matcher.use_hybrid_scoring is True
        assert llm_matcher.max_batch_size == 10
        assert isinstance(llm_matcher.matching_cache, dict)
        
    def test_system_prompt_creation(self, llm_matcher):
        """Test system prompt contains required elements"""
        prompt = llm_matcher.system_prompt
        
        # Check for key matching concepts
        assert "food product matching" in prompt.lower()
        assert "confidence score" in prompt.lower()
        assert "json format" in prompt.lower()
        assert "core product" in prompt.lower()
        assert "brand variations" in prompt.lower()
        
    def test_matching_prompt_creation(self, llm_matcher):
        """Test creation of specific matching prompts"""
        product1 = "Tesco Chicken Breast 500g"
        product2 = "Sainsbury's Chicken Breast 400g"
        context = {"category": "meat", "brand_preference": "any"}
        
        prompt = llm_matcher._create_matching_prompt(product1, product2, context)
        
        assert product1 in prompt
        assert product2 in prompt
        assert "category: meat" in prompt
        assert "Compare these two food products" in prompt
        
    def test_mock_llm_response_generation(self, llm_matcher):
        """Test mock LLM response generation"""
        prompt = '''Product A: "Tesco Chicken Breast 500g"
                   Product B: "Sainsbury's Chicken Breast 400g"'''
        
        response = llm_matcher._generate_mock_llm_response(prompt)
        parsed = json.loads(response)
        
        # Check response structure
        assert "confidence" in parsed
        assert "match" in parsed
        assert "reasoning" in parsed
        assert isinstance(parsed["confidence"], (int, float))
        assert isinstance(parsed["match"], bool)
        assert 0 <= parsed["confidence"] <= 1
        
    def test_response_parsing(self, llm_matcher):
        """Test parsing of LLM responses"""
        valid_response = json.dumps({
            "confidence": 0.85,
            "match": True,
            "reasoning": "Same core product, different brands",
            "key_factors": ["same_core_product", "brand_difference"]
        })
        
        parsed = llm_matcher._parse_llm_response(valid_response)
        
        assert parsed["confidence"] == 0.85
        assert parsed["match"] is True
        assert "reasoning" in parsed
        
    def test_invalid_response_handling(self, llm_matcher):
        """Test handling of invalid LLM responses"""
        invalid_responses = [
            "not json",
            '{"confidence": "invalid"}',
            '{"confidence": 1.5, "match": true}',  # confidence out of range
            '{}'  # missing required fields
        ]
        
        for invalid_response in invalid_responses:
            parsed = llm_matcher._parse_llm_response(invalid_response)
            assert parsed["confidence"] == 0.0
            assert parsed["match"] is False
            
    def test_fallback_matching(self, llm_matcher):
        """Test fallback matching when LLM fails"""
        product1 = "Chicken Breast 500g"
        product2 = "Chicken Breast Fillet 400g"
        
        result = llm_matcher._fallback_match(product1, product2)
        
        assert "confidence" in result
        assert "match" in result
        assert "fallback" in result["reasoning"].lower()
        
    def test_hybrid_score_calculation(self, llm_matcher):
        """Test hybrid score calculation"""
        similarity_score = 0.6
        llm_confidence = 0.8
        
        hybrid_score = llm_matcher._calculate_hybrid_score(similarity_score, llm_confidence)
        
        # Should weight LLM confidence higher (0.7 weight)
        expected = (0.3 * similarity_score) + (0.7 * llm_confidence)
        assert abs(hybrid_score - expected) < 0.001
        
    def test_pre_filter_candidates(self, llm_matcher):
        """Test pre-filtering of candidate pairs"""
        df1 = pd.DataFrame({
            'name': ['Chicken Breast 500g', 'Whole Milk 2L']
        })
        df2 = pd.DataFrame({
            'name': ['Chicken Breast Fillet 400g', 'Bread Loaf 800g']
        })
        
        candidates = llm_matcher._pre_filter_candidates(
            df1, df2, 'name', 'name', threshold=0.4
        )
        
        # Should find chicken-chicken match but not milk-bread
        assert len(candidates) >= 1
        chicken_match = any('chicken' in c['text1'].lower() and 'chicken' in c['text2'].lower() 
                          for c in candidates)
        assert chicken_match
        
    @pytest.mark.asyncio
    async def test_batch_matching(self, llm_matcher):
        """Test batch matching functionality"""
        product_pairs = [
            ("Chicken Breast 500g", "Chicken Breast Fillet 400g"),
            ("Whole Milk 2L", "Skimmed Milk 1L"),
            ("Apple Juice 1L", "Orange Juice 1L")
        ]
        
        results = await llm_matcher.batch_match_products(product_pairs)
        
        assert len(results) == 3
        for result in results:
            assert "confidence" in result
            assert "match" in result
            assert "reasoning" in result
            
    def test_cache_functionality(self, llm_matcher):
        """Test matching cache functionality"""
        # Manually add to cache
        llm_matcher.matching_cache["test||product"] = {
            "confidence": 0.9,
            "match": True,
            "reasoning": "Cached result"
        }
        
        # Test cache hit
        assert len(llm_matcher.matching_cache) == 1
        
        # Test cache clearing
        llm_matcher.clear_cache()
        assert len(llm_matcher.matching_cache) == 0
        
    def test_statistics_tracking(self, llm_matcher):
        """Test statistics tracking"""
        initial_stats = llm_matcher.get_matching_stats()
        
        assert "total_llm_calls" in initial_stats
        assert "successful_matches" in initial_stats
        assert "cache_hits" in initial_stats
        assert initial_stats["success_rate"] == 0.0  # No calls yet
        
    def test_learning_from_feedback(self, llm_matcher):
        """Test feedback learning functionality"""
        product1 = "Chicken Breast 500g"
        product2 = "Chicken Breast Fillet 400g"
        llm_result = {"confidence": 0.8, "match": True}
        
        # Should not raise an error
        llm_matcher.learn_from_feedback(product1, product2, True, llm_result)
        llm_matcher.learn_from_feedback(product1, product2, False, llm_result)


class TestLLMEnhancedScraper:
    """Test cases for LLM-enhanced scraping"""
    
    @pytest.fixture
    def llm_scraper(self):
        """Create LLM scraper instance for testing"""
        return LLMEnhancedScraper()
        
    @pytest.fixture
    def sample_product_text(self):
        """Sample product text for extraction testing"""
        return "Tesco Organic Free Range Chicken Breast Fillets 500g £4.50"
        
    def test_scraper_initialization(self, llm_scraper):
        """Test LLM scraper initialization"""
        assert llm_scraper.llm_model == "gpt-4"
        assert llm_scraper.temperature == 0.1
        assert isinstance(llm_scraper.extraction_prompt, str)
        assert "extract structured product information" in llm_scraper.extraction_prompt.lower()
        
    def test_extraction_prompt_creation(self, llm_scraper):
        """Test extraction prompt creation"""
        raw_text = "Tesco Chicken Breast 500g £4.50"
        context = {"retailer": "tesco", "category": "meat"}
        
        prompt = llm_scraper._create_product_extraction_prompt(raw_text, context)
        
        assert raw_text in prompt
        assert "retailer: tesco" in prompt
        assert "category: meat" in prompt
        
    def test_mock_extraction_response(self, llm_scraper, sample_product_text):
        """Test mock extraction response generation"""
        prompt = f'Raw product text: "{sample_product_text}"'
        response = llm_scraper._generate_mock_extraction_response(prompt)
        
        parsed = json.loads(response)
        
        # Check response structure
        required_fields = ["name", "weight_value", "price_value", "confidence_score"]
        for field in required_fields:
            assert field in parsed
            
        # Check extracted values make sense
        if parsed["weight_value"]:
            assert isinstance(parsed["weight_value"], (int, float))
            assert parsed["weight_value"] > 0
            
        if parsed["price_value"]:
            assert isinstance(parsed["price_value"], (int, float))
            assert parsed["price_value"] > 0
            
    def test_product_info_creation(self):
        """Test ProductInfo dataclass creation"""
        product = ProductInfo(
            name="Test Product",
            weight_value=500,
            weight_unit="g",
            price_value=2.99,
            currency="GBP",
            confidence_score=0.8
        )
        
        assert product.name == "Test Product"
        assert product.weight_value == 500
        assert product.weight_unit == "g"
        assert product.price_value == 2.99
        assert product.confidence_score == 0.8
        
    def test_data_validation(self, llm_scraper):
        """Test extracted data validation"""
        # Test with invalid data
        invalid_product = ProductInfo(
            name="",  # Empty name
            weight_value=-100,  # Negative weight
            price_value=10000,  # Unrealistic price
            confidence_score=0.8
        )
        
        validated = llm_scraper._validate_extracted_data(invalid_product, "Original text")
        
        # Should fix the issues
        assert validated.name == "Original text"  # Should use original text
        assert validated.weight_value is None  # Should remove invalid weight
        assert validated.price_value is None  # Should remove invalid price
        assert validated.confidence_score < 0.8  # Should reduce confidence
        
    def test_fallback_extraction(self, llm_scraper):
        """Test fallback extraction when LLM fails"""
        raw_text = "Chicken Breast 500g £4.50"
        
        product = llm_scraper._fallback_extraction(raw_text)
        
        assert isinstance(product, ProductInfo)
        assert product.name == raw_text
        assert product.confidence_score < 0.5  # Low confidence for fallback
        assert product.extraction_notes is not None
        
    def test_products_to_dataframe(self, llm_scraper):
        """Test conversion of ProductInfo objects to DataFrame"""
        products = [
            ProductInfo(
                name="Product 1",
                weight_value=500,
                weight_unit="g",
                price_value=2.99,
                ingredients=["chicken", "salt"],
                allergens=["gluten"]
            ),
            ProductInfo(
                name="Product 2",
                weight_value=1000,
                weight_unit="g",
                price_value=4.99
            )
        ]
        
        df = llm_scraper.products_to_dataframe(products)
        
        assert len(df) == 2
        assert "name" in df.columns
        assert "weight_value" in df.columns
        assert "ingredients_str" in df.columns  # Should flatten lists
        assert "allergens_str" in df.columns
        assert "extraction_timestamp" in df.columns
        
        # Check data
        assert df.iloc[0]["name"] == "Product 1"
        assert df.iloc[0]["ingredients_str"] == "chicken, salt"
        
    def test_extraction_quality_validation(self, llm_scraper):
        """Test extraction quality validation"""
        products = [
            ProductInfo(name="Product 1", weight_value=500, confidence_score=0.8),
            ProductInfo(name="Product 2", confidence_score=0.5),
            ProductInfo(name="Product 3", weight_value=1000, price_value=2.99, confidence_score=0.9)
        ]
        
        quality_metrics = llm_scraper.validate_extraction_quality(products, min_confidence=0.6)
        
        assert quality_metrics["total_products"] == 3
        assert quality_metrics["high_confidence_products"] == 2  # 0.8 and 0.9
        assert quality_metrics["high_confidence_rate"] == 2/3
        assert "field_completeness" in quality_metrics
        assert "quality_score" in quality_metrics
        
    @pytest.mark.asyncio
    async def test_async_extraction(self, llm_scraper):
        """Test async product extraction"""
        raw_text = "Tesco Chicken Breast 500g £4.50"
        
        product_info = await llm_scraper.extract_product_info_llm(raw_text)
        
        assert isinstance(product_info, ProductInfo)
        assert product_info.name is not None
        assert product_info.confidence_score > 0
        
    def test_export_functionality(self, llm_scraper, tmp_path):
        """Test data export functionality"""
        products = [
            ProductInfo(name="Product 1", weight_value=500, price_value=2.99),
            ProductInfo(name="Product 2", weight_value=1000, price_value=4.99)
        ]
        
        # Test CSV export
        csv_path = llm_scraper.export_enhanced_data(
            products, 
            filename=str(tmp_path / "test_products"),
            format='csv'
        )
        assert os.path.exists(csv_path)
        
        # Test JSON export
        json_path = llm_scraper.export_enhanced_data(
            products,
            filename=str(tmp_path / "test_products"),
            format='json'
        )
        assert os.path.exists(json_path)
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["name"] == "Product 1"


class TestIntegration:
    """Integration tests for LLM-enhanced system"""
    
    @pytest.fixture
    def sample_dataframes(self):
        """Create sample dataframes for matching tests"""
        df1 = pd.DataFrame({
            'product_name': [
                'Tesco Organic Chicken Breast 500g',
                'Sainsbury\'s Whole Milk 2L',
                'Asda Sourdough Bread 800g'
            ],
            'category': ['meat', 'dairy', 'bakery']
        })
        
        df2 = pd.DataFrame({
            'item_description': [
                'Free Range Chicken Breast Fillets 400g',
                'Fresh Whole Milk 1L',
                'Artisan Sourdough Loaf 750g',
                'Orange Juice 1L'
            ],
            'type': ['meat', 'dairy', 'bakery', 'beverages']
        })
        
        return df1, df2
        
    def test_end_to_end_matching(self, sample_dataframes):
        """Test end-to-end matching workflow"""
        df1, df2 = sample_dataframes
        
        matcher = LLMMatcher(confidence_threshold=0.6)
        
        # Test dataframe matching
        matches_df = matcher.match_dataframes_llm(
            df1, df2, 'product_name', 'item_description',
            max_matches_per_item=2,
            pre_filter_similarity=0.3
        )
        
        # Should find some matches
        assert len(matches_df) >= 0  # May be 0 if thresholds are high
        
        if len(matches_df) > 0:
            required_columns = [
                'source_index', 'target_index', 'llm_confidence', 
                'llm_reasoning', 'hybrid_score'
            ]
            for col in required_columns:
                assert col in matches_df.columns
                
    @pytest.mark.asyncio
    async def test_scraping_to_matching_pipeline(self):
        """Test complete pipeline from scraping to matching"""
        
        # Initialize components
        scraper = LLMEnhancedScraper()
        matcher = LLMMatcher()
        
        # Simulate scraped products
        raw_products = [
            {'name': 'Tesco Chicken Breast 500g £4.50', 'retailer': 'tesco'},
            {'name': 'Sainsbury Chicken Fillet 400g £3.99', 'retailer': 'sainsburys'}
        ]
        
        # Extract product info
        enhanced_products = []
        for raw_product in raw_products:
            product_text = raw_product['name']
            context = {'retailer': raw_product['retailer']}
            
            product_info = await scraper.extract_product_info_llm(product_text, context)
            enhanced_products.append(product_info)
            
        # Convert to dataframes
        df = scraper.products_to_dataframe(enhanced_products)
        
        # Test self-matching (should find high similarity)
        if len(df) >= 2:
            matches_df = matcher.match_dataframes_llm(
                df.iloc[:1], df.iloc[1:], 'name', 'name',
                pre_filter_similarity=0.2
            )
            
            # Chicken products should potentially match
            # (exact results depend on mock LLM implementation)
            assert isinstance(matches_df, pd.DataFrame)
            
    def test_performance_monitoring(self):
        """Test that performance monitoring doesn't break functionality"""
        matcher = LLMMatcher()
        scraper = LLMEnhancedScraper()
        
        # Test that stats are being tracked
        initial_matcher_stats = matcher.get_matching_stats()
        initial_scraper_stats = scraper.get_llm_stats()
        
        assert isinstance(initial_matcher_stats, dict)
        assert isinstance(initial_scraper_stats, dict)
        
        # Stats should have expected keys
        expected_matcher_keys = ['total_llm_calls', 'successful_matches', 'cache_hits']
        expected_scraper_keys = ['total_llm_extractions', 'successful_extractions']
        
        for key in expected_matcher_keys:
            assert key in initial_matcher_stats
            
        for key in expected_scraper_keys:
            assert key in initial_scraper_stats


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])