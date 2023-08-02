using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class colorController : MonoBehaviour
{
    MaterialPropertyBlock propertyBlock;
    public Color color;

    void OnValidate()
    {
        //Create a new property block only if it doesn't already exist.
        if (propertyBlock == null)
            propertyBlock = new MaterialPropertyBlock();

        //Gets the 'renderer' component of this object.
        Renderer renderer = GetComponent<Renderer>();

        //You define the color property in the 'property block', whatever is in quotes must match the name of your Shader's color variable.
        propertyBlock.SetColor("_Color", color);

        //The 'propertyBlock' is applied to the renderer, that is: the color.
        renderer.SetPropertyBlock(propertyBlock);
    }

}
