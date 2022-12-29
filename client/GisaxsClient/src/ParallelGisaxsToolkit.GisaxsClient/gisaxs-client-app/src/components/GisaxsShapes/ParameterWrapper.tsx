import Box from '@mui/material/Box/Box';
import TextField from '@mui/material/TextField';
import { Dispatch, SetStateAction } from 'react'

interface ParameterProps {
  parameterName: string
  defaultValue: number
  valueSetter?: Dispatch<SetStateAction<number>>;
}

const ParameterWrapper = (props: ParameterProps) => {

  const readOnly = props.valueSetter === undefined;
  const valueSetter = readOnly ? (e: number) => { } : props.valueSetter
  return (

    <TextField
      label={props.parameterName}
      type="number"
      inputProps={{
        readOnly: readOnly,
        disabled: readOnly
      }}
      onChange={(e) => {
        valueSetter(Number(e.target.value));
      }}
      variant="outlined"
      defaultValue={props.defaultValue}
    />

  );
}

export default ParameterWrapper
