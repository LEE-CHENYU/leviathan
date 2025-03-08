def model_router(model):

    if model == "deepseek":
        provider = "deepseek"
        model_id = "deepseek-chat"
    else:
        provider = "openai"
        model_id = "o3-mini"
        
    return provider, model_id