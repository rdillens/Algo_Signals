class Product:
    def __init__(self, df, output, pattern_list):
        self.data = []
        self.df = df
        self.out = output
        self.pattern_list = pattern_list

    def show_df(self):
        df = self.df.copy()
        self.out.clear_output()

