def get_on_fit_config(p1_noise, p2_noise):
    """
    Ez a függvény generálja azt a konfigurációt, amit a szerver 
    minden kör elején leküld a klienseknek.
    """
    def fit_config_fn(server_round: int):
        # Itt adjuk át a mátrixodból jövő zajszinteket
        return {
            "p1_noise": p1_noise,
            "p2_noise": p2_noise,
            "current_round": server_round,
        }
    
    return fit_config_fn