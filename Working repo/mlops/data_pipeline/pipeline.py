from dagster import pipeline, solid

@solid
def extract_data(context):
    # Logic to extract data from a source
    source_data = extract_data_from_source()
    context.log.info('Data extracted successfully.')
    return source_data

@solid
def transform_data(context, source_data):
    # Logic to transform the extracted data
    transformed_data = transform_data_logic(source_data)
    context.log.info('Data transformed successfully.')
    return transformed_data

@solid
def load_data(context, transformed_data):
    # Logic to load the transformed data to a destination
    load_data_to_destination(transformed_data)
    context.log.info('Data loaded successfully.')

@pipeline
def data_pipeline():
    transformed_data = transform_data(extract_data())
    load_data(transformed_data)

