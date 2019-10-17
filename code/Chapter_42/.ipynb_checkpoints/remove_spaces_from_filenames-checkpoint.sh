for oldname in *
do
  newname=`echo $oldname | sed -e 's/ /_/g'`
  if [ "$newname" = "$oldname" ]
  then
    continue
  fi
  if [ -e "$newname" ]
  then
    echo Skipping "$oldname", because "$newname" exists
  else
    mv "$oldname" "$newname"
  fi
done