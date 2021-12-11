import { createTheme, ThemeProvider} from '@material-ui/core/styles';
import CssBaseline from "@material-ui/core/CssBaseline";
import Content from "./Content"

function App()
{
    const darkTheme = createTheme({
        palette: {
            type: 'dark',
        }
    });

    return (
        <ThemeProvider theme={darkTheme}>
            <CssBaseline/>
            <Content/>
        </ThemeProvider>
    );
}

export default App;