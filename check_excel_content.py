#!/usr/bin/env python3
"""Check Excel content to verify function breakdown"""

import pandas as pd

# Read the Function Breakdown sheet
try:
    df = pd.read_excel('Lambda_Document_Processor_Performance_Analysis.xlsx', sheet_name='Function Breakdown')
    print("Function Breakdown Sheet Contents:")
    print("=" * 50)
    print(df.to_string(index=False))
    print(f"\nTotal functions listed: {len(df)}")
    
    # Check if all expected functions are there
    expected_functions = [
        'get_models()',
        'fitz.open() + doc.load_page()',
        'should_enhance_image()',
        'enhance_image_for_ocr()',
        'ocr.predict() - PaddleOCR',
        'structure_pipeline.predict() - PPStructureV3',
        'create_spatial_index()',
        'find_matching_structure()',
        'bbox_overlap()',
        'combine_ocr_and_structure()',
        'is_quality_content()',
        'build_hierarchical_structure()',
        'create_semantic_chunks()',
        'create_chunk()'
    ]
    
    actual_functions = df['Function'].tolist()
    print(f"\nExpected key functions found:")
    for func in expected_functions:
        if func in actual_functions:
            print(f"✅ {func}")
        else:
            print(f"❌ {func}")
            
except Exception as e:
    print(f"Error reading Excel: {e}")