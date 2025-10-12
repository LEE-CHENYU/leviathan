def model_router(model):

    if model == "deepseek":
        provider = "deepseek"
        model_id = "deepseek-chat"
    else:
        provider = "openai"
        model_id = "gpt-5"
        
    return provider, model_id