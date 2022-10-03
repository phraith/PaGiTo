
import Box from "@mui/material/Box/Box";
import Button from "@mui/material/Button"

const Logout = () => {
  const removeToken = () => {
    localStorage.setItem('apiToken', '')
    window.dispatchEvent(new Event("storage"));
  };

  return (
    <Box>
      <Button color="inherit" onClick={removeToken}>Logout</Button>
    </Box>
  );
};

export default Logout;
