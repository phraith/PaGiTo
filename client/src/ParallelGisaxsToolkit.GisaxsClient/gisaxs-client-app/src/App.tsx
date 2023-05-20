import Simulation from "./components/Simulation/Simulation";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import Fitting from "./components/Fitting/Fitting";
import { Fragment } from "react";
import Jobs from "./components/Jobs/Jobs";
import { ThemeOptions, ThemeProvider, createTheme } from '@mui/material/styles';

const App = () => {


  const themeOptions: ThemeOptions = {
    palette: {
      mode: 'dark',
      background: {
        default: '#2d2d30',
        paper: '#2E3B55', // your color
      },
      primary: {
        main: '#2E3B55',
      },
      secondary: {
        main: '#dc114b',
      },
    },
  };
  const theme = createTheme(themeOptions);
  return (
    <Fragment>
      <ThemeProvider theme={theme}>
        <BrowserRouter>
          <Routes>
            <Route path="/" element={<Simulation />} />
            <Route path="simulation" element={<Simulation />} />
            <Route path="fitting" element={<Fitting />} />
            <Route path="jobs" element={<Jobs />} />
          </Routes>
        </BrowserRouter>
      </ThemeProvider>
    </Fragment>

  );
};

export default App;
