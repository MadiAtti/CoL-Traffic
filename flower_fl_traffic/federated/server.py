def get_on_fit_config(**kwargs):
    """
    Univerzális konfiguráció generáló. 
    Bármilyen kulcsszavas érvet (kwargs) kap, azt minden körben leküldi a klienseknek.
    """
    def fit_config_fn(server_round: int):
        # Alapértelmezett adatok, amik mindig kellenek (pl. aktuális kör)
        config = {"current_round": server_round}
        
        # Hozzáadjuk a kísérlet-specifikus paramétereket
        config.update(kwargs)
        
        return config
    
    return fit_config_fn